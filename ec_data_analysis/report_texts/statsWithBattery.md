Now we add a hypothetical battery to the community. Every member has the ability to charge the battery (similar to selling to a community member) and discharge from the battery (similar to buying from a community member). The charging/discharging privileges are allocated in an analogous way to the market clearing algorithm.

The battery is only allowed to discharge to 15% of its maximal capacity and charge up to 85% of its maximal capacity. The initial capacity of the battery is 15% of the initial capacity. Note that this initial energy storage may not be discharged by the members.

The conversion loss for each battery transaction is 5% and the static loss is 0.1% per hour. The c-rate is 0.5, meaning that in each timestep, the battery may be (dis)charged by a fraction of at most c-rate * timestep/hour. For example if the timestep is 15 minutes and the c-rate is 0.5, the battery may be (dis)charged by at most 12.5% of its full capacity per timestep.

The required battery capacity for full communal self-consumption refers to the battery capacity required to be able to put the entire supply left-over from the market into the battery. Since the hypothetical battery is of this size, the percentage of supply put to battery plus the supply sold on the market consequently add up to 100%.
