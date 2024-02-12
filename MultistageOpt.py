from Optimierung2 import createNetwork,getScenarios,extra_functionality,extra_postprocessing,setScenarios
from gesammelteVorhersage import GesammelteVorhersage
from Daten.Zeitreihe import ladeVorverarbeiteteDaten
import pandas as pd
import time
import warnings
import os
import sys
from multiprocessing import Pool

warnings.filterwarnings("ignore", category=FutureWarning) 

def multistageOpti(startDay,days=4):
    os.makedirs("Results/Logs",exist_ok=True)
    sys.stdout = open("Results/Logs/"+str(os.getpid()) + ".out", "w")
    sys.stderr = open("Results/Logs/"+str(os.getpid()) + "_error.out", "w")

    n_clusters = 0
    overlap = 12

    stopDate = pd.Timestamp(startDay).replace(hour=23) + pd.Timedelta(days=days)
    snapshots = pd.date_range(startDay,stopDate,freq="H")
    gv = GesammelteVorhersage(disp=False)

    def copyBids(nFrom,nTo):
        bidsC = nFrom.global_constraints_t.bid_capacity.copy()
        bidsP = nFrom.global_constraints_t.bid_price.copy()
        nTo.global_constraints_t.bid_capacity = bidsC
        nTo.global_constraints_t.bid_price = bidsP

    df = ladeVorverarbeiteteDaten("1H",filled=True).loc[snapshots]
    aFRR_pos = df[["pos_aFRR_[EURO/MW]"]]
    aFRR_pos.columns = [0]
    aFRR_neg = df[["neg_aFRR_[EURO/MW]"]]
    aFRR_neg.columns = [0]
    Da = df[["Da_[EUR/MWh]"]]
    Da.columns = [0]
    Q_fern = df[["Q_FernW_[MW]"]]
    Q_fern.columns = [0]
    p_1 = pd.Series({0:1})
    actual = {"aFRR_pos":aFRR_pos,"aFRR_neg":aFRR_neg,"Da":Da,"Q_fern":Q_fern}
    def replaceWithActual(preds,bids,snapshots):
        for bid in bids:
            preds[bid] = (actual[bid].loc[snapshots].copy(),p_1)
    def addBottomLine(preds,bids):
        for bid in bids:
            preds[bid][0].loc[:,n_clusters] = -500
            preds[bid][1].loc[n_clusters]=1e-6

    timing = {f"Stage {i}":[] for i in range(1,4)}

    def saveResults(networks):
        costs = {}
        os.makedirs(f"Results/Timing",exist_ok=True)
        os.makedirs(f"Results/TotalCost",exist_ok=True)
        for i,n in enumerate(networks):
            # Name unter welchem das Ergebnis abgelegt werden soll
            name = f"{startDay}-{days}-{n_clusters}/Stage-{i+1}"
            os.makedirs(f"Results/{name}",exist_ok=True)

            n.export_to_csv_folder(f"Results/{name}")
            n.priceLevels.reset_index(drop=True).to_csv(f"Results/{name}/priceLevels.csv")
            n.p_scenarios.reset_index(drop=True).to_csv(f"Results/{name}/p_scenarios.csv")
            n.global_constraints_t.bid_capacity.reset_index(drop=True).to_csv(f"Results/{name}/bid_capacity.csv")
            n.global_constraints_t.bid_price.reset_index(drop=True).to_csv(f"Results/{name}/bid_price.csv")
            costs[f"Stage {i+1}"] = n.costs
        pd.DataFrame.from_dict(timing).to_csv(f"Results/Timing/Timing-{startDay}-{days}-{n_clusters}.csv")
        pd.DataFrame(costs).to_csv(f"Results/TotalCost/TotalCost-{startDay}-{days}-{n_clusters}.csv")

    def run():
        # Erstellen des Netzwerks für Stage 1
        predsStage1 = gv.getClusteredForecast(9,snapshots[0],daysToForecast=2,n_samples=1000,n_clusters=n_clusters)
        addBottomLine(predsStage1,["Da"])
        n = createNetwork(predsStage1,snapshots)

        # Erstellen des Netzwerks für Stage 2
        predsStage2 = gv.getClusteredForecast(12,snapshots[0],daysToForecast=2,n_samples=1000,n_clusters=n_clusters)
        replaceWithActual(predsStage2,["aFRR_pos","aFRR_neg"],snapshots)
        addBottomLine(predsStage2,["Da"])
        n2 = createNetwork(predsStage2,snapshots)

        # Erstellen des Netzwerks für Stage 3
        predsStage3 = gv.getClusteredForecast(23,snapshots[0],daysToForecast=2,n_samples=1000,n_clusters=n_clusters)
        replaceWithActual(predsStage3,["Da","aFRR_pos","aFRR_neg","Q_fern"],snapshots)
        n3 = createNetwork(predsStage3,snapshots)

        for day in range(days):
            start = day*24
            stop = (day+1)*24+overlap

            if start != 0:
                # Speicherfüllungen aus letzter optimierung übernehmen
                waterLast = n3.stores_t.e.at[snapshots[start-1],"water_S0000"]
                n.stores["e_initial"] = waterLast
                n2.stores["e_initial"] = waterLast
                n3.stores["e_initial"] = waterLast
                batLast = n3.storage_units_t.state_of_charge.at[snapshots[start-1],"bat_S0000"]
                n.storage_units["state_of_charge_initial"] = batLast
                n2.storage_units["state_of_charge_initial"] = batLast
                n3.storage_units["state_of_charge_initial"] = batLast

            print("### STAGE 1 ###")
            print(f"Day: {snapshots[start]}")
            startTime = time.time()
            predsStage1 = gv.getClusteredForecast(9,snapshots[start],daysToForecast=2,n_samples=1000,n_clusters=n_clusters)
            addBottomLine(predsStage1,["Da"])
            setScenarios(n,predsStage1)
            n.lopf(snapshots[start:stop], solver_name = "gurobi", pyomo=True,extra_functionality=extra_functionality,extra_postprocessing=extra_postprocessing)
            endTime = time.time()
            timing["Stage 1"].append(endTime-startTime)

            print("### STAGE 2 ###")
            print(f"Day: {snapshots[start]}")
            startTime = time.time()
            predsStage2 = gv.getClusteredForecast(12,snapshots[start],daysToForecast=2,n_samples=1000,n_clusters=n_clusters)
            replaceWithActual(predsStage2,["aFRR_pos","aFRR_neg"],snapshots[start:stop])
            copyBids(n,n2)
            # Durch das setzen von Da auf NaN wird diese nicht fixiert und wieder als Entscheidungsvariable optimiert
            n2.global_constraints_t.bid_price.loc[snapshots[start:stop],"Da"] = float("nan")
            n2.global_constraints_t.bid_capacity.loc[snapshots[start:stop],"Da"] = float("nan")
            addBottomLine(predsStage2,["Da"]) # Gebot auf Minestpreis ermöglichen -> Abgabe eines Angebots, welches immer angenommen wird -> Vermeidung Infeasable durch Nichteinhaltung der aFRR_pos
            setScenarios(n2,predsStage2)
            n2.lopf(snapshots[start:stop], solver_name = "gurobi", pyomo=True,extra_functionality=extra_functionality,extra_postprocessing=extra_postprocessing)
            endTime = time.time()
            timing["Stage 2"].append(endTime-startTime)

            print("### STAGE 3 ###")
            print(f"Day: {snapshots[start]}")
            startTime = time.time()
            predsStage3 = gv.getClusteredForecast(23,snapshots[start],daysToForecast=2,n_samples=1000,n_clusters=n_clusters)
            replaceWithActual(predsStage3,["Da","aFRR_pos","aFRR_neg","Q_fern"],snapshots[start:stop])
            copyBids(n2,n3)
            setScenarios(n3,predsStage3)
            n3.lopf(snapshots[start:stop], solver_name = "gurobi", pyomo=True,extra_functionality=extra_functionality,extra_postprocessing=extra_postprocessing)
            endTime = time.time()
            timing["Stage 3"].append(endTime-startTime)

            # Zwischenspeichern
            saveResults([n,n2,n3])

    # for n_clusters in range(5,0,-1):
    n_clusters = 5
    timing = {f"Stage {i}":[] for i in range(1,4)}
    run()

if __name__ == "__main__":
    startDays = ["2019-01-08"]
    # startDays = ["2019-01-08","2019-04-01","2019-07-01"]
    with Pool(processes=3) as pool:
        pool.map(multistageOpti,startDays)