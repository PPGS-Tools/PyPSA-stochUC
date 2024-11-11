from pypsa import Network
from pypsa.descriptors import get_switchable_as_dense
from linopy import LinearExpression
from linopy.expressions import merge
import pandas as pd
import numpy as np
import xarray as xr

def add_reserve_market_to_component_attrs(components,component_attrs):
    components.loc["aFRR_Market"] = ["aFRR_market","German secondary balancing reserve Market",np.nan]
    component_attrs["aFRR_Market"] = pd.DataFrame(columns=["type", "unit", "default", "description", "status"])
    component_attrs["aFRR_Market"].index.name = "attribute"
    component_attrs["aFRR_Market"].loc["name"] = ['string', np.nan, np.nan,'Name of the market','Input (required)']
    component_attrs["aFRR_Market"].loc["marginal_cost_pos"] = ['static or series', 'currency/MW/h', 0,'Marginal cost of positive balancing reserve of 1 MW for 1 hour.', 'Input (optional)']
    component_attrs["aFRR_Market"].loc["marginal_cost_neg"] = ['static or series', 'currency/MW/h', 0,'Marginal cost of negative balancing reserve of 1 MW for 1 hour.', 'Input (optional)']
    component_attrs["aFRR_Market"].loc["include_bids"] = ['boolean', np.nan, False,'Switch to include bidding for this market.','Input (optional)']
    component_attrs["aFRR_Market"].loc["bid_capacity"] = ['series', 'MW/h', 0,'Capacity of resulting market bids for balancing reserve','Output']


def add_reserve_to_component_attrs(component_attrs):
    comps = ("Generator","StorageUnit")
    for comp in comps:
        component_attrs[comp].loc["reserve_pos_max"] = ["static or series","MW",0.0,"Positive reservable power limit","Input (optional)"]
        component_attrs[comp].loc["reserve_neg_max"] = ["static or series","MW",0.0,"Negative reservable power limit","Input (optional)"]
        component_attrs[comp].loc["reserve_pos"] = ["series","MW",0.0,"Positive power reserve","Output"]
        component_attrs[comp].loc["reserve_neg"] = ["series","MW",0.0,"Negative power reserve","Output"]


def _add_reserve_to_model(n: Network)->tuple[LinearExpression,LinearExpression]:
    """Add reserve constraints and variables to the model

    Parameters
    ----------
    n : Network
        PyPSA network. The fuction adds reserve variables for all generators and storage units.

    Returns
    -------
    tuple[LinearExpression,LinearExpression]
        Total positive and negative reserve power.
    """
    m = n.model
    total_reserve_pos = 0
    total_reserve_neg = 0
    res_pos_g, res_neg_g = _add_generator_reserve_variables(n, m)

    total_reserve_pos += res_pos_g.sum("Generator-fix")
    total_reserve_neg += res_neg_g.sum("Generator-fix")

    res_pos_su, res_neg_su = _add_storage_unite_reserve_variables(n, m)

    total_reserve_pos += res_pos_su.sum("StorageUnit-fix")
    total_reserve_neg += res_neg_su.sum("StorageUnit-fix")

    return total_reserve_pos, total_reserve_neg


def _add_storage_unite_reserve_variables(n, m):
    """
    Adds reserve variables for storage units to the model.

    Parameters:
    ----------
    n (Network): The network object containing the storage units.
    m (Model): The model object to which the reserve variables will be added.

    Returns:
    ----------
    tuple: A tuple containing the positive and negative reserve variables for storage units.
           If there are no non-extendable storage units, returns (0, 0).

    Notes:
    ----------
    - This function adds positive and negative reserve variables for storage units.
    - It modifies the constraints related to storage unit dispatch, storage, and state of charge.
    """
    sus = n.get_non_extendable_i("StorageUnit")
    if not sus.empty:
        coords = m.constraints["StorageUnit-fix-p_store-upper"].indexes
        res_max = get_switchable_as_dense(n, "StorageUnit", "reserve_pos_max",inds=coords["StorageUnit-fix"])
        res_max.columns.name = "StorageUnit-fix"
        res_pos_su = m.add_variables(name="StorageUnit-reserve_pos",coords=coords,lower=0,upper=res_max)
        m.constraints["StorageUnit-fix-p_dispatch-upper"].lhs+=res_pos_su
        m.constraints["StorageUnit-fix-p_store-lower"].lhs-=res_pos_su
        m.constraints["StorageUnit-fix-state_of_charge-lower"].lhs-=res_pos_su.shift(snapshot=1)

        res_max = get_switchable_as_dense(n, "StorageUnit", "reserve_neg_max",inds=coords["StorageUnit-fix"])
        res_max.columns.name = "StorageUnit-fix"
        res_neg_su = m.add_variables(name="StorageUnit-reserve_neg",coords=coords,lower=0,upper=res_max)
        m.constraints["StorageUnit-fix-p_dispatch-lower"].lhs-=res_neg_su
        m.constraints["StorageUnit-fix-p_store-upper"].lhs+=res_neg_su
        m.constraints["StorageUnit-fix-state_of_charge-upper"].lhs += res_neg_su.shift(snapshot=1)
        return res_pos_su,res_neg_su
    else:
        return 0,0


def _add_generator_reserve_variables(n, m):
    """
    Adds reserve variables for generators to the model.

    Parameters:
    n (Network): The network object containing generator data.
    m (Model): The optimization model to which reserve variables are added.

    Returns:
    tuple: A tuple containing the positive and negative reserve variables for generators.
           If there are no non-extendable generators, returns (0, 0).
    """
    gens = n.get_non_extendable_i("Generator")
    if not gens.empty:
        coords = m.constraints["Generator-fix-p-upper"].indexes
        res_max = get_switchable_as_dense(n, "Generator", "reserve_pos_max",inds=coords["Generator-fix"])
        res_max.columns.name = "Generator-fix"
        res_pos_g = m.add_variables(name="Generator-reserve_pos",coords=coords,lower=0,upper=res_max)
        m.constraints["Generator-fix-p-upper"].lhs+=res_pos_g

        res_max = get_switchable_as_dense(n, "Generator", "reserve_neg_max",inds=coords["Generator-fix"])
        res_max.columns.name = "Generator-fix"
        res_neg_g = m.add_variables(name="Generator-reserve_neg",coords=coords,lower=0,upper=res_max)
        m.constraints["Generator-fix-p-lower"].lhs-=res_neg_g
        return res_pos_g,res_neg_g
    else:
        return 0,0

def add_reserve_to_model(n):
    """
    Adds reserve components to the given PyPSA network model.
    Used if reserve market is not included in the model.

    Parameters:
    n (pypsa.Network): The PyPSA network model to which reserve components will be added.

    Returns:
    None
    """
    _add_reserve_to_model(n)

def add_reserve_market_to_model(n: Network):
    """
    Adds the reserve market to the given PyPSA network model.

    This function integrates the reserve market into the PyPSA network model by
    adding necessary variables, constraints, and modifying the objective function
    based on the reserve market data.

    Parameters:
    -----------
    n : Network
        The PyPSA network object to which the reserve market will be added.

    Notes:
    ------
    - The function first calls `add_reserve_to_model` to get the total reserve
      requirements.
    - If the reserve market does not include bids, the marginal costs are directly
      subtracted from the objective function.
    - If the reserve market includes bids, it sets up variables for bid capacity,
      calculates accepted bids, and adds constraints to ensure the total accepted
      capacity is reserved by generators and storage units.
    - Additional constraints are added to ensure bid capacities are consistent
      over 4-hour blocks.
    """
    m = n.model
    snapshots = n.snapshots

    total_reserve_pos, total_reserve_neg = _add_reserve_to_model(n)
    if not n.aFRR_market.include_bids.any():
        marginal_cost_pos = n.aFRR_market.marginal_cost_pos
        marginal_cost_neg = n.aFRR_market.marginal_cost_neg
        m.objective-=(marginal_cost_pos*total_reserve_pos).sum()+(marginal_cost_neg*total_reserve_neg).sum()
    else:
        markets = n.aFRR_market[n.aFRR_market.include_bids].index
        markets.name = "market"
        bid_price = get_aFRR_prices()
        aFRRPriceLevels = bid_price.coords["pricelevel"]
        prod = bid_price.coords["prod"]
        bid_capacity = m.add_variables(name="aFRR-bid_capacity",coords=[snapshots,markets,prod,aFRRPriceLevels],lower=0)

        marginal_cost_pos = get_switchable_as_dense(n, "aFRR_Market", "marginal_cost_pos",inds=markets).unstack().to_xarray()
        marginal_cost_pos = marginal_cost_pos.rename({"level_0":"market"})
        marginal_cost_neg = get_switchable_as_dense(n, "aFRR_Market", "marginal_cost_neg",inds=markets).unstack().to_xarray()
        marginal_cost_neg = marginal_cost_neg.rename({"level_0":"market"})
        marginal_cost = xr.concat([marginal_cost_pos,marginal_cost_neg],prod)
        accepted = bid_price<=marginal_cost
        total_accepted_capacity = (accepted*bid_capacity).sum(["pricelevel","market"])
        # revenue = (total_accepted_capacity * marginal_cost).sum()
        revenue = (accepted*bid_capacity*bid_price).sum()
        total_pos = total_accepted_capacity.sel({"prod":"pos"},drop=True)
        m.add_constraints(total_pos == total_reserve_pos,name="aFRR-bid_capacity_constraint_pos")
        total_neg = total_accepted_capacity.sel({"prod":"neg"},drop=True)
        m.add_constraints(total_neg == total_reserve_neg,name="aFRR-bid_capacity_constraint_neg")

        groups = (bid_capacity.indexes["snapshot"].hour%4).values
        nBlocks = (groups==3).sum()
        groups[nBlocks*4:] = -1
        lhs = [
            bid_capacity[groups==0] - bid_capacity[groups==1],
            bid_capacity[groups==1] - bid_capacity[groups==2],
            bid_capacity[groups==2] - bid_capacity[groups==3],

        ]
        lhs = merge(lhs,join="outer")
        m.add_constraints(lhs==0,name="aFRR-bid_capacity_constraint_4h_blocks")

        m.objective-=revenue

def get_aFRR_bids(n: Network)->pd.DataFrame:
    bids_pos = n.model.variables["aFRR-bid_capacity"].sel(prod="pos",market="aFRR").solution.to_pandas()
    bids_neg = n.model.variables["aFRR-bid_capacity"].sel(prod="neg",market="aFRR").solution.to_pandas()
    bids_price= get_aFRR_prices().sel(market="aFRR")
    bids_pos.columns = bids_price.sel(prod="pos")
    bids_neg.columns = bids_price.sel(prod="neg")
    bids = pd.concat([bids_pos,bids_neg],axis=1,keys=["pos","neg"],names=["prod"])
    # only keep the first bid for every 4-hour block
    bids = bids[bids.index.hour%4==0]
    return bids

def get_aFRR_prices()->xr.DataArray:
    aFRRPriceLevels = pd.RangeIndex(0,11,name="pricelevel")
    bid_price = xr.DataArray(np.linspace(0,50,len(aFRRPriceLevels)),coords=[aFRRPriceLevels],name="price")
    return bid_price.expand_dims(market=["aFRR"],prod=["pos","neg"])