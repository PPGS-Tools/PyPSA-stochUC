import pypsa 
import numpy as np 
import pandas as pd
from itertools import product

import matplotlib.pyplot as plt
import matplotlib as mpl
from pyomo.environ import *


from pypsa.descriptors import (
    get_switchable_as_dense,
    allocate_series_dataframes
)
from pypsa.opt import (
    LConstraint,
    LExpression,
    free_pyomo_initializers,
    l_constraint,
)

RANGE_PRICEINDEX = True

# def extendComponents():
#     #Fügt Component für Bids hinzu

#     # from https://github.com/PyPSA/PyPSA/blob/acf39ab6d8dda31b7e19460e9bcbfdf3debf26f1/examples/new_components/add_components_simple.py
#     # take a copy of the components pandas.DataFrame
#     override_components = pypsa.components.components.copy()

#     # Pass it the list_name, description and component type.
#     override_components.loc["Markets"] = [
#         "markets",
#         "Component to output merket bids.",
#         np.nan,
#     ]   
        
#     return override_components

def extendComponentAttrs():
    #Erweitert die Components comp und Storage Unit um eine Variable zur Vorhaltung der Reserveleistung

    attr = pypsa.descriptors.Dict({k: v.copy() for k, v in pypsa.components.component_attrs.items()})
    comps = ("Generator","StorageUnit")

    for comp in comps:

        attr[comp].loc["r_pos_max"] = \
            ["static or series","MW",0.0,"Maximum reserve requirement","Input (optional)"]
        
        attr[comp].loc["r_neg_max"] = \
            ["static or series","MW",0.0,"Maximum reserve requirement","Input (optional)"]
        
        attr[comp].loc["r_pos"] = \
            ["series","MW",0.0,"Active reserve at bus","Output"]

        attr[comp].loc["r_neg"] = \
            ["series","MW",0.0,"Active reserve at bus","Output"]
        
    # attr["Markets"] = pd.DataFrame(
    #     columns=["type", "unit", "default", "description", "status"]
    # )
    # attr["Markets"].loc["name"] = \
    #     ["string",np.nan,np.nan,"Unique name","Input (required)"]
    
    attr["GlobalConstraint"].loc["bid_capacity"] = \
        ["series","MW(h)",0.0,"Capacity of resulting market bids","Output"]
    
    attr["GlobalConstraint"].loc["bid_price"] = \
        ["series","Euro/capacity",0.0,"Resulting market bids","Output"]
        
    return attr


def createNetwork(preds,snapshots):


    n = pypsa.Network(snapshots = snapshots, override_component_attrs=extendComponentAttrs())

    #Hier werden die maximal vorhaltbaren Reserven in positive und negative Richtung begrenzt
    n.total_reserve_pos= 100
    n.total_reserve_neg= 100
    n.totalCost = 0
    n.costs = []

    setScenarios(n,preds)

    S=n.scenarios

    addBuses(n,S)
    addKraftwerk(n,S)
    addLinks(n,S)
    addBatterySystem(n,S)
    addWaermespeicher(n,S)
    addDayAheadMarkt(n,S)
    
    return n

def addBuses(n,s):
    n.madd("Bus","buselec"+s, carrier = "AC")
    n.madd("Bus","busb"+s, carrier = "Li-ion")
    n.madd("Bus","busgas"+s, carrier = "gas")
    n.madd("Bus","busheat"+s, carrier = "heat")
    n.madd("Bus","busstore"+s, carrier = "water")
    n.madd("Bus","busloss"+s, carrier = "heat")

def addKraftwerk(n,s):
     n.madd("Generator","genloadgas"+s, bus = "busgas"+s, p_nom = 60, carrier = "gas",
        p_max_pu=1,
        p_min_pu=36/60,
        r_pos_max=n.total_reserve_pos,r_neg_max=n.total_reserve_neg,
        ramp_limit_up=0.01,ramp_limit_down=0.01,
        committable=True,
        marginal_cost=-27.51428571)
     n.madd("Generator","loss"+s, bus = "busloss"+s, p_nom=10000, p_max_pu=0, p_min_pu=-1, marginal_cost=0)

def addLinks(n,s):
    n.madd("Link","chpelec"+s, bus0 = "busgas"+s, bus1 = "buselec"+s, efficiency = 1,p_nom=13.5,ramp_limit_up=0.1,ramp_limit_down=0.1,p_min_pu=0,p_max_pu=1)
    n.madd("Link","chpheat"+s, bus0 = "busgas"+s, bus1 = "busheat"+s,efficiency=1, p_nom = 30)
    n.madd("Link","chploss"+s, bus0 = "busgas"+s, bus1 = "busloss"+s, efficiency = 1, p_nom = 60,p_min_pu=0,p_max_pu=1)

def addBatterySystem(n,s):
    #aus https://juser.fz-juelich.de/record/908382/files/Energie_Umwelt_577.pdf
    effbat=0.9797 #wrzl(0,96)
    standinglossbat=0
    n.madd("Link","eleccharge"+s, bus0 = "buselec"+s, bus1 = "busb"+s, p_nom = 6, efficiency = effbat)
    n.madd("Link","elecdischarge"+s, bus0 = "busb"+s, bus1 = "buselec"+s, p_nom = 6, efficiency = effbat)

    n.madd("StorageUnit","bat"+s, bus="busb"+s, state_of_charge_initial=3,p_nom=6,standing_loss = standinglossbat,p_min_pu=-1,
        r_pos_max=n.total_reserve_pos,r_neg_max=n.total_reserve_neg)

def addWaermespeicher(n,s):
    #aus https://juser.fz-juelich.de/record/908382/files/Energie_Umwelt_577.pdf
    effwatertank=0.9899 #wrzl(0,98)
    standinglossheat=0.0001
    n.madd("Store","water"+s, bus="busstore"+s, e_nom_min=0, e_nom=50, e_initial=25,standing_loss=standinglossheat)

    n.madd("Link","heatcharge"+s, bus0="busheat"+s, bus1="busstore"+s, p_nom=10, efficiency=effwatertank)
    n.madd("Link","heatdischarge"+s, bus0="busstore"+s, bus1="busheat"+s, p_nom=10, efficiency=effwatertank)

    n.madd("Load","loadheat"+s, bus = "busheat"+s, p_set = n.Q_fern, carrier = "heat")
    n.madd("Generator","penaltyQ"+s, bus = "busheat"+s, p_max_pu=1, p_min_pu=0,p_nom= 10000,marginal_cost = 1000, carrier = "heat")

def addDayAheadMarkt(n,s):
    n.madd("Generator","dagen"+s,bus="buselec"+s, p_max_pu=1, p_min_pu=-1,p_nom= 10000,marginal_cost = n.da)
    n.madd("Generator","penalty"+s, bus = "buselec"+s, p_max_pu=1, p_min_pu=0,p_nom= 10000,marginal_cost = 1000)
    # n.madd("Generator","penaltyNeg"+s, bus = "buselec"+s, p_max_pu=0, p_min_pu=-1,p_nom= 10000,marginal_cost = -1000)

def setScenarios(n,preds):
    scenarios,p_scenarios = getScenarios(preds,{"aFRR_pos":"aFRRpos","aFRR_neg":"aFRRneg","Da":"dagen","Q_fern":"loadheat"})
    snapshots = scenarios["Da"].index
    nScenarios = len(p_scenarios)
    print(f"{nScenarios} Szenarios werden erstellt ...")
    S = pd.Index([f"_S{s:04d}" for s in range(nScenarios)])
    n.scenarios = S
    if not hasattr(n, 'p_scenarios'):
        # init p_scenarios for all snapshots
        first = n.snapshots[0]
        last = n.snapshots[-1]
        dayIndex = pd.date_range(first,last,freq="D")
        n.p_scenarios = pd.DataFrame(index=dayIndex,columns=p_scenarios.keys(),dtype=float)
        n.da = pd.DataFrame(index=n.snapshots,columns=scenarios["Da"].columns,dtype=float)
        n.aFRR_pos = pd.DataFrame(index=n.snapshots,columns=scenarios["aFRR_pos"].columns,dtype=float)
        n.aFRR_neg = pd.DataFrame(index=n.snapshots,columns=scenarios["aFRR_neg"].columns,dtype=float)
        n.Q_fern = pd.DataFrame(index=n.snapshots,columns=scenarios["Q_fern"].columns,dtype=float)
    first = snapshots[0]
    last = snapshots[-1]
    dayIndex = pd.date_range(first,last,freq="D")
    n.p_scenarios.loc[dayIndex] = pd.Series(p_scenarios).values   
        
    n.da.loc[snapshots] = scenarios["Da"].astype(float)
    n.aFRR_pos.loc[snapshots] = scenarios["aFRR_pos"].astype(float).clip(0)
    n.aFRR_neg.loc[snapshots] = scenarios["aFRR_neg"].astype(float).clip(0)
    n.Q_fern.loc[snapshots] = scenarios["Q_fern"].astype(float).clip(lower=0,upper=29.605900) # Dies ist der Maximalwert des Datensatzes. Bei überschreitung wird das Problem infeasible

    if not RANGE_PRICEINDEX:
        priceLevels = {}
        priceLevels["Da"] = preds["Da"][0]
        priceLevels["aFRR_pos"] = preds["aFRR_pos"][0].clip(0) # Negative Preise sind hier nicht möglich
        priceLevels["aFRR_neg"] = preds["aFRR_neg"][0].clip(0) # Negative Preise sind hier nicht möglich
        if not hasattr(n,"priceLevels"):
            n.priceLevels = pd.DataFrame(index=n.snapshots,columns=pd.concat(priceLevels,1).columns)
        priceLevels = pd.concat(priceLevels,1)
        n.priceLevels.loc[snapshots,priceLevels.columns] = priceLevels

    else:
        if not hasattr(n,"priceLevels"):
            # Für aFRR_pos, aFRR_neg und werden 10 Preisniveaus erstellt
            columns = pd.MultiIndex.from_product([["aFRR_pos","aFRR_neg","Da"],range(10)])
            n.priceLevels = pd.DataFrame(index=n.snapshots,columns=columns)
            n.priceLevels.loc[:,"Da"] = np.concatenate([[-500],np.linspace(-100,100,10)])
            n.priceLevels.loc[:,"aFRR_pos"] = np.concatenate([np.linspace(0,10,6),np.linspace(15,30,4)])
            n.priceLevels.loc[:,"aFRR_neg"] = np.concatenate([np.linspace(0,10,6),np.linspace(15,30,4)])

    if len(n.generators.index)>0:
        n.generators_t.marginal_cost.loc[snapshots,"dagen"+S] = n.da
        n.loads_t.p_set.loc[snapshots,"loadheat"+S] = n.Q_fern

def getScenarios(preds,componentNames):
    """
    Generator to access all Scenarios
    """

    keys = preds.keys()

    def getScenarioCombinations(keys):
        """
        Returns a Generator for the base Scenario Combinations
        """
        baseScenarios = []
        for key in keys:
            pred = preds[key][0]
            baseScenarios.append(pred.columns)
        return product(*baseScenarios)

    def getScenarioByCombination(combination:tuple):
        """
        Returns the Scenario, associated with the given combination. \\
        The order of the Tuple has to match the order of "keys"
        """
        scenario = {}
        p = 1
        for i,key in enumerate(keys):
            col = combination[i]
            pred = preds[key][0].loc[:,col]
            p_i = preds[key][1].loc[col]
            p *= p_i
            scenario[key] = pred
        return (scenario,p)

    combinations = getScenarioCombinations(keys)
    scenarios = {key:{} for key in keys}
    p_scenarios = {}
    for s,combination in enumerate(combinations):
            scenario,p = getScenarioByCombination(combination)
            p_scenarios[f"_S{s:04d}"] = p
            for key in keys:
                compName = componentNames[key]
                scenarios[key][f"{compName}_S{s:04d}"] = scenario[key]
    for key in keys:
        scenarios[key] = pd.DataFrame.from_dict(scenarios[key])
    return scenarios,p_scenarios

# from gesammelteVorhersage import GesammelteVorhersage
# if __name__ == "__main__":
#     preds = GesammelteVorhersage().getClusteredForecast(10,"2019-01-19")
#     scen = getScenarios(preds,{"aFRR_pos":"aFRRpos","aFRR_neg":"aFRRneg","Da":"dagen","Q_fern":"loadheat"})
#     n = createNetwork(scen)
#     print(n)


def pos_reserve_constraints(network,snapshots):
    #Hier wird die positive SRL definiert

    #Hier werden die Bounds definiert also von 0 bis R_max, für SU und Gen 
    # Gl 4.5
    def gen_r_nom_bounds_pos(model, gen_name,snapshot):
        return (0,network.generators.at[gen_name,"r_pos_max"])

    def stor_r_nom_bounds_pos(model, stor_name, snapshot):
        return (0,network.storage_units.at[stor_name,"r_pos_max"])

    #Hier werden die Variablen gespeichert
    fixed_committable_gens_i = network.generators.index[~network.generators.p_nom_extendable & network.generators.committable]

    p_max_pu = get_switchable_as_dense(network, 'Generator', 'p_max_pu', snapshots)
    
    #Hier werden die Variablen initialisert
    network.model.generator_r_pos = Var(network.generators.index, snapshots, domain=Reals, bounds=gen_r_nom_bounds_pos)
    free_pyomo_initializers(network.model.generator_r_pos)

    #Gespeichert
    sus = network.storage_units
    fixed_sus_i = sus.index[~ sus.p_nom_extendable] # Only committable storage units allowed
    stor_p_max_pu = get_switchable_as_dense(network, 'StorageUnit', 'p_max_pu', snapshots)

    #Initialisiert
    network.model.storage_units_r_pos = Var(list(fixed_sus_i),snapshots,domain=Reals,bounds=stor_r_nom_bounds_pos)
    free_pyomo_initializers(network.model.storage_units_r_pos)

    #SU Gleichungen, auch zu finden in der BA 
    #Energie SU 
    # Gl 4.10 geändert
    stor_p_r_upper_soc_pos = {}
    for su in list(fixed_sus_i):
        # Umgang mit previous state of carge aus opf.py
        for i, sn in enumerate(snapshots):
            stor_p_r_upper_soc_pos[su, sn] = [[], ">=", 0.0]
            if i == 0:
                previous_state_of_charge = network.storage_units.at[su, "state_of_charge_initial"]
                stor_p_r_upper_soc_pos[su,sn][2] -= previous_state_of_charge
            else:
                previous_state_of_charge = network.model.state_of_charge[su, snapshots[i - 1]]
                stor_p_r_upper_soc_pos[su,sn][0].append((1,previous_state_of_charge))

            stor_p_r_upper_soc_pos[su, sn][0].extend([
                (1,network.model.storage_p_store[su,sn]),
                (-1,network.model.storage_p_dispatch[su,sn]),
                (-1,network.model.storage_units_r_pos[su,sn]), 
            ])
    l_constraint(network.model,"stor_p_r_upper_soc_pos",stor_p_r_upper_soc_pos, fixed_sus_i,snapshots)

    
    #leistung SU 
    # Gl 4.12 jedoch um store ergänzt
    stor_p_r_upper_dis_pos = {(stor,sn) :
                        [[(1,network.model.storage_p_dispatch[stor,sn]),                         #pDischarge
                          (-1,network.model.storage_p_store[stor,sn]),                           #TW Hat gefehlt
                        (1,network.model.storage_units_r_pos[stor,sn]),                                 #pRes
                        ],
                        "<=",stor_p_max_pu.at[sn,stor]*network.storage_units.p_nom[stor]]
                        for stor in list(fixed_sus_i) for sn in snapshots}

    l_constraint(network.model,"stor_p_r_upper_dis_pos",stor_p_r_upper_dis_pos, list(fixed_sus_i),snapshots)

    #Leistungsbegrenzung Gen
    # Gl. 4.12
    gen_p_r_upper_pos = {(gen,sn) :
                [[(1,network.model.generator_p[gen,sn]),
                (1,network.model.generator_r_pos[gen,sn]),
                (-p_max_pu.at[sn, gen]*network.generators.p_nom[gen],network.model.generator_status[gen,sn])
                ],
                "<=",0.]
                for gen in fixed_committable_gens_i for sn in snapshots}
    l_constraint(network.model, "generator_p_r_upper_pos", gen_p_r_upper_pos, list(fixed_committable_gens_i), snapshots)

def neg_reserve_constraints(network,snapshots):

    #Hier wird im Grunde genommen das gleiche wie bei der positiven Reserve gemacht, die Bounds sind auch hier von 0 bis R_max
    #Wichtig ist, dass nur Generatoren, die comittable sind Reserve vorhalten können 
    def gen_r_nom_bounds_neg(model, gen_name,snapshot):
        return (0,network.generators.at[gen_name,"r_neg_max"])

    def stor_r_nom_bounds_neg(model, stor_name, snapshot):
        return (0,network.storage_units.at[stor_name,"r_neg_max"])

    fixed_committable_gens_i = network.generators.index[~network.generators.p_nom_extendable & network.generators.committable]


    p_min_pu = get_switchable_as_dense(network, 'Generator', 'p_min_pu', snapshots)

    network.model.generator_r_neg = Var(list(network.generators.index), snapshots, domain=Reals, bounds=gen_r_nom_bounds_neg)
    free_pyomo_initializers(network.model.generator_r_neg)

    
    sus = network.storage_units
    fixed_sus_i = sus.index[~ sus.p_nom_extendable] # Only committable storage units allowed
    stor_p_min_pu = get_switchable_as_dense(network, 'StorageUnit', 'p_min_pu', snapshots)

    network.model.storage_units_r_neg = Var(list(fixed_sus_i),snapshots,domain=Reals,bounds=stor_r_nom_bounds_neg)
    free_pyomo_initializers(network.model.storage_units_r_neg)

    # Gl. 4.15 (geändert)
    stor_p_r_lower_soc_neg = {}
    for su in list(fixed_sus_i):
        # Umgang mit previous state of carge aus opf.py
        for i, sn in enumerate(snapshots):
            stor_p_r_lower_soc_neg[su, sn] = [[], "<=", network.storage_units.p_nom[su]]
            if i == 0:
                previous_state_of_charge = network.storage_units.at[su, "state_of_charge_initial"]
                stor_p_r_lower_soc_neg[su,sn][2] -= previous_state_of_charge
            else:
                previous_state_of_charge = network.model.state_of_charge[su, snapshots[i - 1]]
                stor_p_r_lower_soc_neg[su,sn][0].append((1,previous_state_of_charge))

            stor_p_r_lower_soc_neg[su, sn][0].extend([
                (1,network.model.storage_p_store[su,sn]),
                (-1,network.model.storage_p_dispatch[su,sn]),
                (+1,network.model.storage_units_r_neg[su,sn]), 
            ])
    l_constraint(network.model,"stor_p_r_lower_soc_neg",stor_p_r_lower_soc_neg, fixed_sus_i,snapshots)

    # Gl 4.16, jedoch um dispatch ergänzt
    stor_p_r_upper_dis_neg = {(stor,sn) :
                        [[(1,network.model.storage_p_store[stor,sn]),                         #pDischarge
                          (-1,network.model.storage_p_dispatch[stor,sn]),                      #TW pCharge
                        (1,network.model.storage_units_r_neg[stor,sn]),                                 #pRes
                        ],
                        "<=",-1.0*stor_p_min_pu.at[sn,stor]*network.storage_units.p_nom[stor]]
                        for stor in list(fixed_sus_i) for sn in snapshots}

    l_constraint(network.model,"stor_p_r_upper_dis_neg",stor_p_r_upper_dis_neg, list(fixed_sus_i),snapshots)

    # statt Gl. 4.17 & 4.18 Muss hier nicht chpelec verwendet werden?
    gen_p_r_lower_neg = {(gen,sn) :
                [[(1,network.model.generator_p[gen,sn]),
                (-1,network.model.generator_r_neg[gen,sn]),
                (-p_min_pu.at[sn, gen]*network.generators.p_nom[gen],network.model.generator_status[gen,sn])
                ],
                ">=",0.]
                for gen in fixed_committable_gens_i for sn in snapshots}
    l_constraint(network.model, "generator_p_r_lower_neg", gen_p_r_lower_neg, list(fixed_committable_gens_i), snapshots)


def top_iso_fuel_line(model, snapshot,s):
    return ( 24.73506702 + 2.71904565*(model.link_p["chpelec"+s, snapshot]+model.generator_r_pos['genloadgas'+s,snapshot])
            + 0.61319625*model.link_p["chpheat"+s, snapshot] 
            <= model.generator_p['genloadgas'+s,snapshot])

def cor_res(model, snapshot,s):
    # Muss eingeführt werden um die negative Reserve richtig darzustellen
    return (model.link_p["chpelec"+s, snapshot]>=model.generator_r_neg['genloadgas'+s,snapshot])

def four_hour_comittment_gen(model,snapshot4H,offset,s,sign):
    freq = snapshot4H.freq / 4
    if sign == "pos":
        r = model.generator_r_pos
    else:
        r = model.generator_r_neg    
    return r["genloadgas"+s,snapshot4H+freq*offset] == r["genloadgas"+s,snapshot4H+freq*(1+offset)]

def four_hour_comittment_bat(model,snapshot4H,offset,s,sign):
    freq = snapshot4H.freq / 4
    if sign == "pos":
        r = model.storage_units_r_pos
    else:
        r = model.storage_units_r_neg   
    return r["bat"+s,snapshot4H+freq*offset] == r["bat"+s,snapshot4H+freq*(1+offset)]

def stochasticOpt(network,snapshots):
    def isFixed(bidName):
        if not bidName in network.global_constraints_t.bid_capacity.columns:
            return False
        if not bidName in network.global_constraints_t.bid_price.columns:
            return False
        if network.global_constraints_t.bid_capacity.loc[snapshots,bidName].isna().any().any():
            return False
        if network.global_constraints_t.bid_price.loc[snapshots,bidName].isna().any().any():
            return False
        return True
    
    if isFixed("Da"):
        columns = network.global_constraints_t.bid_price[["Da"]].columns
        network.priceLevels[columns] = network.global_constraints_t.bid_price[columns]
        unwantedColumns = network.priceLevels["Da"].columns.difference(network.global_constraints_t.bid_price["Da"].columns)
        unwantedColumns = pd.MultiIndex.from_product([["Da"],unwantedColumns])
        network.priceLevels.drop(columns=unwantedColumns,inplace=True)
    if isFixed("aFRR_pos"): 
        columns = network.global_constraints_t.bid_price[["aFRR_pos"]].columns
        network.priceLevels[columns] = network.global_constraints_t.bid_price[columns]
        unwantedColumns = network.priceLevels["aFRR_pos"].columns.difference(network.global_constraints_t.bid_price["aFRR_pos"].columns)
        unwantedColumns = pd.MultiIndex.from_product([["aFRR_pos"],unwantedColumns])
        network.priceLevels.drop(columns=unwantedColumns,inplace=True)
    if isFixed("aFRR_neg"):
        columns = network.global_constraints_t.bid_price[["aFRR_neg"]].columns
        network.priceLevels[columns] = network.global_constraints_t.bid_price[columns]
        unwantedColumns = network.priceLevels["aFRR_neg"].columns.difference(network.global_constraints_t.bid_price["aFRR_neg"].columns)
        unwantedColumns = pd.MultiIndex.from_product([["aFRR_neg"],unwantedColumns])
        network.priceLevels.drop(columns=unwantedColumns,inplace=True)

    network.priceLevels = network.priceLevels[["Da","aFRR_pos","aFRR_neg"]] # Order columns of PriceLevels
    priceLevels = network.priceLevels

    #init Price index
    network.model.J = RangeSet(0,len(priceLevels["Da"].columns)-1)
    J = network.model.J
    network.model.K = RangeSet(0,len(priceLevels["aFRR_pos"].columns)-1)
    K = network.model.K
    network.model.L = RangeSet(0,len(priceLevels["aFRR_neg"].columns)-1)
    L = network.model.L

    S = network.scenarios

    #Init Variables
    network.model.Da_bid= Var(snapshots,S,J,domain=Reals,bounds=(-500,3000))
    free_pyomo_initializers(network.model.Da_bid)

    network.model.aFRR_pos_bid= Var(snapshots,S,K, domain=NonNegativeReals)
    free_pyomo_initializers(network.model.aFRR_pos_bid)

    network.model.aFRR_neg_bid= Var(snapshots,S,L, domain=NonNegativeReals)
    free_pyomo_initializers(network.model.aFRR_neg_bid)

    network.model.Da_dispatch= Var(snapshots,S,domain=Reals)
    free_pyomo_initializers(network.model.Da_dispatch)

    network.model.aFRR_pos_dispatch= Var(snapshots,S,K, domain=NonNegativeReals)
    free_pyomo_initializers(network.model.aFRR_pos_dispatch)

    network.model.aFRR_neg_dispatch= Var(snapshots,S,L, domain=NonNegativeReals)
    free_pyomo_initializers(network.model.aFRR_neg_dispatch)

    # fix bids
    if isFixed("Da"):
        for s in S:
            for j in J:
                for sn in snapshots:
                    network.model.Da_bid[sn,s,j].fix(network.global_constraints_t.bid_capacity.at[sn,("Da",j)])
    if isFixed("aFRR_pos"):
        for s in S:
            for k in K:
                for sn in snapshots:
                    network.model.aFRR_pos_bid[sn,s,k].fix(network.global_constraints_t.bid_capacity.at[sn,("aFRR_pos",k)])
    if isFixed("aFRR_neg"):
        for s in S:
            for l in L:
                for sn in snapshots:
                    network.model.aFRR_neg_bid[sn,s,l].fix(network.global_constraints_t.bid_capacity.at[sn,("aFRR_neg",l)])
                    
    
    # Bids must be equal for all scenarios
    Da_bids_equal = {}
    for sn in snapshots:
        for i, s in enumerate(S[:-1]):
            for j in J:
                    Da_bids_equal[sn,s,j] = [[], "==", 0.0]
                    Da_bids_equal[sn,s,j][0] =[
                        ( 1,network.model.Da_bid[sn,s,j]),
                        (-1,network.model.Da_bid[sn,S[i+1],j])
                    ]
    l_constraint(network.model,"Da_bids_equal",Da_bids_equal,snapshots,S[:-1],J)
    aFRR_pos_bids_equal = {}
    for sn in snapshots:
        for i, s in enumerate(S[:-1]):
            for k in K:
                    aFRR_pos_bids_equal[sn,s,k] = [[], "==", 0.0]
                    aFRR_pos_bids_equal[sn,s,k][0] =[
                        ( 1,network.model.aFRR_pos_bid[sn,s,k]),
                        (-1,network.model.aFRR_pos_bid[sn,S[i+1],k])
                    ]
    l_constraint(network.model,"aFRR_pos_bids_equal",aFRR_pos_bids_equal,snapshots,S[:-1],K)
    aFRR_neg_bids_equal = {}
    for sn in snapshots:
        for i, s in enumerate(S[:-1]):
            for l in L:
                    aFRR_neg_bids_equal[sn,s,l] = [[], "==", 0.0]
                    aFRR_neg_bids_equal[sn,s,l][0] =[
                        ( 1,network.model.aFRR_neg_bid[sn,s,l]),
                        (-1,network.model.aFRR_neg_bid[sn,S[i+1],l])
                    ]
    l_constraint(network.model,"aFRR_neg_bids_equal",aFRR_neg_bids_equal,snapshots,S[:-1],L)

    sus = network.storage_units.index[~ network.storage_units.p_nom_extendable] # Only committable storage units allowed
    gens = network.generators.index[~network.generators.p_nom_extendable & network.generators.committable]


    # Sum up all accepted bids (dispatch) per scenario and timestep and make them equal to the szenarios reserves
    aFRR_pos_dispatch_sum = {}
    for sn in snapshots:
        for s in S:
            aFRR_pos_dispatch_sum[sn,s] = [[], "==", 0.0]

            for k in K:
                aFRR_pos_dispatch_sum[sn,s][0].append((-1,network.model.aFRR_pos_dispatch[sn,s,k]))

            gens_s = gens[gens.str.endswith(s)]
            sus_s = sus[sus.str.endswith(s)]
            for gen in gens_s:
                aFRR_pos_dispatch_sum[sn,s][0].append((1,network.model.generator_r_pos[gen,sn]))
            for su in sus_s:
                aFRR_pos_dispatch_sum[sn,s][0].append((1,network.model.storage_units_r_pos[su,sn]))
                        
    l_constraint(network.model,"aFRR_pos_dispatch_sum",aFRR_pos_dispatch_sum,snapshots,S)

    aFRR_neg_dispatch_sum = {}
    for sn in snapshots:
        for s in S:
            aFRR_neg_dispatch_sum[sn,s] = [[], "==", 0.0]

            for l in L:
                aFRR_neg_dispatch_sum[sn,s][0].append((-1,network.model.aFRR_neg_dispatch[sn,s,l]))

            gens_s = gens[gens.str.endswith(s)]
            sus_s = sus[sus.str.endswith(s)]
            for gen in gens_s:
                aFRR_neg_dispatch_sum[sn,s][0].append((1,network.model.generator_r_neg[gen,sn]))
            for su in sus_s:
                aFRR_neg_dispatch_sum[sn,s][0].append((1,network.model.storage_units_r_neg[su,sn]))
                        
    l_constraint(network.model,"aFRR_neg_dispatch_sum",aFRR_neg_dispatch_sum,snapshots,S)

    # Sum up all accepted da bids to dispatch
    gamma = {}
    for sn in snapshots:
        for s in S:
            for j in J:
                gamma[sn,s,j] = 1 if priceLevels.loc[sn,"Da"].loc[j] <= network.generators_t.marginal_cost.at[sn,"dagen"+s] else 0

    Da_dispatch_sum = {}
    for sn in snapshots:
        for s in S:
            Da_dispatch_sum[sn,s] = [[], "==", 0.0]
            Da_dispatch_sum[sn,s][0].append((-1,network.model.Da_dispatch[sn,s]))
            for j in J:
                Da_dispatch_sum[sn,s][0].append((gamma[sn,s,j],network.model.Da_bid[sn,s,j]))
                        
    l_constraint(network.model,"Da_dispatch_sum",Da_dispatch_sum,snapshots,S)

    if not isFixed("Da"):
        # Dieser Constraint setzt Gebote, deren Preise höher als alle Szenarien sind auf 0
        # Dieser Constraint gilt jedoch nur solange die Gebote nicht bereits vor der Optimierung festliegen (fixed)
        Da_bid_unused ={}
        for sn in snapshots:
            for j in J:
                if not any([gamma[sn,s,j] for s in S]):
                    Da_bid_unused[sn,j] = [(1,network.model.Da_bid[sn,s,j])],"==",0
                else:
                    Da_bid_unused[sn,j] = [],"==",0

        l_constraint(network.model,"Da_bid_unused",Da_bid_unused,snapshots,J)


    beta = {}
    for sn in snapshots:
        for s in S:
            for k in K:
                beta[sn,s,k] = 1 if priceLevels.loc[sn,"aFRR_pos"].loc[k] <= network.aFRR_pos.loc[sn,"aFRRpos"+s] else 0

    aFRR_pos_dispatch_accepted ={}
    for sn in snapshots:
        for s in S:
            for k in K:
                aFRR_pos_dispatch_accepted[sn,s,k] = [[],"==",0]
                aFRR_pos_dispatch_accepted[sn,s,k][0] = [
                    (1,network.model.aFRR_pos_dispatch[sn,s,k]),
                    (-beta[sn,s,k],network.model.aFRR_pos_bid[sn,s,k])
                ]

    l_constraint(network.model,"aFRR_pos_dispatch_accepted",aFRR_pos_dispatch_accepted,snapshots,S,K)

    beta = {}
    for sn in snapshots:
        for s in S:
            for l in L:
                beta[sn,s,l] = 1 if priceLevels.loc[sn,"aFRR_neg"].loc[l] <= network.aFRR_neg.loc[sn,"aFRRneg"+s] else 0

    aFRR_neg_dispatch_accepted ={}
    for sn in snapshots:
        for s in S:
            for l in L:
                aFRR_neg_dispatch_accepted[sn,s,l] = [[],"==",0]
                aFRR_neg_dispatch_accepted[sn,s,l][0] = [
                    (1,network.model.aFRR_neg_dispatch[sn,s,l]),
                    (-beta[sn,s,l],network.model.aFRR_neg_bid[sn,s,l])
                ]

    l_constraint(network.model,"aFRR_neg_dispatch_accepted",aFRR_neg_dispatch_accepted,snapshots,S,L)

    # Map Da dispatch to Da Generator
    Da_dispatch_gen = {}
    for sn in snapshots:
        for s in S:
            Da_dispatch_gen[sn,s] = [[], "==", 0.0]
            Da_dispatch_gen[sn,s][0] =[
                ( 1,network.model.Da_dispatch[sn,s]),
                ( 1,network.model.generator_p["dagen"+s,sn])
            ]
    l_constraint(network.model,"Da_dispatch_gen",Da_dispatch_gen,snapshots,S)

    network.model.alpha = Var(snapshots, domain=Binary)
    free_pyomo_initializers(network.model.alpha)

    Da_sell = {}
    for sn in snapshots:
        for s in S:
            for j in J:
                Da_sell[sn,s,j] = [[], "<=", 0.0]
                Da_sell[sn,s,j][0] =[
                    ( 1,network.model.Da_bid[sn,s,j]),
                    (-10000,network.model.alpha[sn])
                ]
    l_constraint(network.model,"Da_sell",Da_sell,snapshots,S,J)

    Da_buy = {}
    for sn in snapshots:
        for s in S:
            for j in J:
                Da_buy[sn,s,j] = [[], "<=", 10000]
                Da_buy[sn,s,j][0] =[
                    (-1,network.model.Da_bid[sn,s,j]),
                    ( 10000,network.model.alpha[sn])
                ]
    l_constraint(network.model,"Da_buy",Da_buy,snapshots,S,J)
                



def redefine_linear_objective(network, snapshots):
    
    #Hier wird die Objective Funktion (also Kostenfunktion) um die Therme der SRL erweitert.
    #Auch hier müssen die Terme immer manuell hinzugefügt werden, was noch bearbeitet gehört.
    #neg und pos Reserve sind gleich aufgebaut, nur die Preise sind natürlich verschieden

    oldObjCoefs = network.model.objective.expr.linear_coefs 
    oldObjVars = network.model.objective.expr.linear_vars 

    
    for snapshot in snapshots:    
    
        oldObjCoefs.append(-network.aFRRpos.at[snapshot,"aFRRpos_S0000"])
        oldObjVars.append(network.model.generator_r_pos['genloadgas_S0000',snapshot])
        
        oldObjCoefs.append(-network.aFRRpos.at[snapshot,"aFRRpos_S0000"])
        oldObjVars.append(network.model.storage_units_r_pos['bat_S0000',snapshot])
        
        oldObjCoefs.append(-network.aFRRneg.at[snapshot,"aFRRneg_S0000"])
        oldObjVars.append(network.model.generator_r_neg['genloadgas_S0000',snapshot])
        
        oldObjCoefs.append(-network.aFRRneg.at[snapshot,"aFRRneg_S0000"])
        oldObjVars.append(network.model.storage_units_r_neg['bat_S0000',snapshot])
        
    
    
    oldObjConst = network.model.objective.expr.constant
    oldObjSense = network.model.objective.sense
    
    index = range(len(oldObjCoefs))
    network.model.del_component(network.model.objective)
    network.model.objective = Objective(expr=sum(oldObjVars[i]*oldObjCoefs[i] for i in index)+oldObjConst, sense=oldObjSense)
    # Das print könnte entfernt werden, zeigt aber immer ganz gut wie die Kostenfunktion aufgebaut wird
    print(network.model.objective.expr.to_string())

def stochasticObjective(network,snapshots):
    S = network.scenarios
    day = snapshots[0]
    cost =-sum([
        network.p_scenarios.loc[day][s]*sum([
            -network.generators.at["genloadgas_S0000","marginal_cost"]*network.model.generator_p["genloadgas"+s,sn]
            +network.generators_t.marginal_cost.at[sn,"dagen"+s]*network.model.Da_dispatch[sn,s]
            +sum([
              network.priceLevels.loc[sn,"aFRR_pos"].iloc[k]*network.model.aFRR_pos_dispatch[sn,s,k]
            for k in network.model.K])
            +sum([
              network.priceLevels.loc[sn,"aFRR_neg"].iloc[l]*network.model.aFRR_neg_dispatch[sn,s,l]
            for l in network.model.L])
            -network.generators.at["penaltyQ"+s,"marginal_cost"]*network.model.generator_p["penaltyQ"+s,sn]
            -network.generators.at["penalty"+s,"marginal_cost"]*network.model.generator_p["penalty"+s,sn]
            # -network.generators.at["penaltyNeg_S0000","marginal_cost"]*network.model.generator_p["penaltyNeg"+s,sn]
        for sn in snapshots]) 
    for s in S])
    network.model.del_component(network.model.objective)
    objective = Objective(expr = cost,sense=minimize)
    objective.construct()
    # print(objective.expr.to_string().replace("+","\n +"))
    network.model.objective = objective

def extra_functionality(n, snapshots):
    # Rangeset als Iterator für Scenarios

    neg_reserve_constraints(n,snapshots)
    pos_reserve_constraints(n,snapshots)

    n.model.top_iso_fuel_line = Constraint(snapshots,n.scenarios, rule = top_iso_fuel_line)
    n.model.cor_res = Constraint(snapshots,n.scenarios, rule = cor_res)
    assert len(snapshots)%4==0
    n.model.four_hour_comittment_gen = Constraint(snapshots[::4],range(3),n.scenarios,("pos","neg"),rule = four_hour_comittment_gen)
    n.model.four_hour_comittment_bat = Constraint(snapshots[::4],range(3),n.scenarios,("pos","neg"),rule = four_hour_comittment_bat)
    # redefine_linear_objective(n, snapshots)
    stochasticOpt(n,snapshots)
    stochasticObjective(n,snapshots)

# # iis code from https://groups.google.com/g/pypsa/c/UDFnQAyILWg
# solver_parameters = "ResultFile=model.ilp" # write an ILP file to print the IIS
# n.model = None
# n.model = pypsa.opf.network_lopf_build_model(n,n.snapshots,formulation="kirchhoff")
# extra_functionality(n,n.snapshots)
# opt = pypsa.opf.network_lopf_prepare_solver(n, solver_name="gurobi")
# n.results=opt.solve(n.model, options_string=solver_parameters,tee=True)
# n.results.write()

def extra_postprocessing(network, snapshots, duals):
    """
    Extracts the new results and adds them to the pypsa network.
    Results are written to:
    network.generators_t.r_pos
    network.generators_t.r_neg
    network.storage_units_t.r_pos
    network.storage_units_t.r_neg
    network.global_constraints_t.bid_capacity
    network.global_constraints_t.bid_price
    """
    allocate_series_dataframes(
        network,
        {
            "Generator": ["r_pos","r_neg"],
            "StorageUnit": ["r_pos","r_neg"],
        },
    )
    if not len(network.global_constraints_t.bid_capacity):
        allocate_series_dataframes(network,{"GlobalConstraint": ["bid_capacity","bid_price"]})

    # from opf.py
    def clear_indexedvar(indexedvar):
        for v in indexedvar._data.values():
            v.clear()

    def get_values(indexedvar, free=True):
        s = pd.Series(indexedvar.get_values(), dtype=float)
        if free:
            clear_indexedvar(indexedvar)
        return s

    def set_from_series(df, series):
        df.loc[snapshots] = series.unstack(0).reindex(columns=df.columns)

    def set_from_series(df, series):
        df.loc[snapshots] = series.unstack(0).reindex(columns=df.columns)

    model = network.model

    if len(network.generators):
        set_from_series(network.generators_t.r_pos, get_values(model.generator_r_pos))
        set_from_series(network.generators_t.r_neg, get_values(model.generator_r_neg))

    if len(network.storage_units):
        set_from_series(network.storage_units_t.r_pos, get_values(model.storage_units_r_pos))
        set_from_series(network.storage_units_t.r_neg, get_values(model.storage_units_r_neg))

    prices  = [model.Da_bid,model.aFRR_pos_bid,model.aFRR_neg_bid]
    bid_price = network.priceLevels.copy()
    columns = bid_price.columns

    bid_capacity = pd.concat([get_values(price).xs("_S0000",level=1).unstack() for price in prices],axis=1)
    bid_capacity.columns = columns

    network.global_constraints_t.bid_capacity = network.global_constraints_t.bid_capacity.reindex(columns=columns)
    network.global_constraints_t.bid_price = network.global_constraints_t.bid_price.reindex(columns=columns)
    network.global_constraints_t.bid_capacity.loc[snapshots,columns] = bid_capacity.loc[snapshots]
    network.global_constraints_t.bid_price.loc[snapshots,columns] = bid_price.loc[snapshots]

    df = network.global_constraints_t.bid_capacity
    df[(-1e-3 < df) & (df < 0)] = 0 # Runde Rechenfehler kleiner 0 auf 0

    network.costs.append(network.objective)
