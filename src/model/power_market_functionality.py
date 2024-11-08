from pypsa import Network
from pypsa.descriptors import get_switchable_as_dense
from linopy import LinearExpression
from linopy.expressions import merge
import pandas as pd
import numpy as np
import xarray as xr

def add_power_market_to_component_attrs(component_attrs):
    component_attrs["Generator"].loc["is_market"] = ['boolean', np.nan, False,'Switch if this is a power bidding market.','Input (optional)']
    # component_attrs["Generator"].loc["include_bids"] = ['boolean', np.nan, False,'Switch to include bidding for this market.','Input (optional)']
    component_attrs["Generator"].loc["bid_capacity"] = ['series', 'MW/h', 0,'Capacity of resulting market bids','Output']
    component_attrs["Generator"].loc["bid_price"] = ['series', 'currency/MWh/h', 0,'Resulting market bids','Output']

def add_power_market_to_model(n):
    """
    Adds the power market to the given model.
    This function integrates the power market into the provided model by adding
    variables and constraints related to day-ahead bid capacities.
    Parameters:
    n : pypsa.Network
        The network object containing the model and snapshots.
    Notes:
    - The function identifies market generators that include bids.
    - It creates variables for day-ahead bid capacities across different price levels.
    - It calculates the bid prices and compares them to the marginal cost.
    - Constraints are added to ensure that the accepted bid capacity matches the 
      generator's power output for the identified market generators.
    """

    m = n.model
    snapshots = n.snapshots

    markets = n.generators.index[n.generators.is_market & n.generators.include_bids]
    bid_price_da = get_da_prices(n)
    daPriceLevels = bid_price_da.coords["pricelevel"]
    bid_capacity = m.add_variables(name="DayAhead-bid_capacity",coords=[snapshots,markets,daPriceLevels])

    marginal_cost = get_switchable_as_dense(n, "Generator", "marginal_cost",inds=markets).unstack().to_xarray()
    accepted = bid_price_da<=marginal_cost
    total_accepted_capacity = (accepted*bid_capacity).sum("pricelevel")
    generate_accepted_capacity_constraint = -m.variables["Generator-p"].sel(Generator=markets) == total_accepted_capacity
    m.add_constraints(generate_accepted_capacity_constraint,name="DayAhead-bid_capacity_constraint")

    # Revenue is already handled by the Generator component

def get_da_bids(n: Network)->pd.DataFrame:
    bids = n.model.variables["DayAhead-bid_capacity"].sel(Generator="da").solution.to_pandas()
    bids_price= get_da_prices(n).sel(Generator="da")
    bids.columns = bids_price
    return bids

def get_da_prices(n: Network)->xr.DataArray:
    daPriceLevels = pd.RangeIndex(0,11,name="pricelevel")
    bid_price = xr.DataArray(np.linspace(0,100,len(daPriceLevels)),coords=[daPriceLevels],name="price")
    markets = n.generators.index[n.generators.is_market & n.generators.include_bids]
    return bid_price.expand_dims(Generator=markets)