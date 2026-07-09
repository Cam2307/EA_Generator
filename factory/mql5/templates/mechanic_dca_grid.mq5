//@SECTION INPUTS
input group "=== Execution: DCA Grid ==="
input double {IN_grid_step_points} = {P_grid_step_points}; // Grid step (points)
input double {IN_lot_multiplier}   = {P_lot_multiplier};   // Grid lot multiplier
input int    {IN_max_levels}       = {P_max_levels};       // Max grid levels
input double {IN_basket_tp_points} = {P_basket_tp_points}; // Basket TP from avg (points)
//@SECTION GLOBALS
int    g_grid_levels = 0;
double g_grid_extreme = 0.0;
int    g_grid_direction = 0;
//@SECTION FUNCTIONS
void OpenEntry(const ENUM_ORDER_TYPE type)
  {
   const double lots = NormalizeLots(InpLots);
   SendMarketOrder(type, lots, 0.0, 0.0, "EAF_G0");
   if(CountOurPositions() > 0)
     {
      g_grid_levels = 1;
      g_grid_direction = (type == ORDER_TYPE_BUY) ? 1 : -1;
      g_grid_extreme = (type == ORDER_TYPE_BUY)
         ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
         : SymbolInfoDouble(_Symbol, SYMBOL_BID);
      g_tm_trades_today++;
     }
  }

bool GridState(int &direction, int &count, double &extreme_entry, double &avg_entry)
  {
   direction = 0;
   count = 0;
   extreme_entry = 0.0;
   double lot_sum = 0.0, weighted = 0.0;

   if(IsHedgingAccount())
     {
      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         const ulong ticket = PositionGetTicket(i);
         if(ticket == 0 || !PositionSelectByTicket(ticket) || !IsOurPosition())
            continue;
         const double entry = PositionGetDouble(POSITION_PRICE_OPEN);
         const double vol   = PositionGetDouble(POSITION_VOLUME);
         const int dir = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 1 : -1;
         if(direction == 0)
           {
            direction = dir;
            extreme_entry = entry;
           }
         if(dir == direction)
           {
            count++;
            lot_sum  += vol;
            weighted += entry * vol;
            if((direction > 0 && entry < extreme_entry) ||
               (direction < 0 && entry > extreme_entry))
               extreme_entry = entry;
           }
        }
     }
   else
     {
      // netting: one aggregated position — use tracked grid state
      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         const ulong ticket = PositionGetTicket(i);
         if(ticket == 0 || !PositionSelectByTicket(ticket) || !IsOurPosition())
            continue;
         direction = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 1 : -1;
         count = g_grid_levels;
         extreme_entry = g_grid_extreme;
         lot_sum = PositionGetDouble(POSITION_VOLUME);
         weighted = PositionGetDouble(POSITION_PRICE_OPEN) * lot_sum;
         break;
        }
     }

   if(count == 0 || lot_sum <= 0.0)
      return(false);
   avg_entry = SafeDiv(weighted, lot_sum);
   return(true);
  }

void ManagePositions()
  {
   int direction = 0, count = 0;
   double extreme = 0.0, avg = 0.0;
   if(!GridState(direction, count, extreme, avg))
     {
      g_grid_levels = 0;
      g_grid_direction = 0;
      return;
     }

   const double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   const double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   const double price = (direction > 0) ? bid : ask;

   const double target = avg + direction * {IN_basket_tp_points} * _Point;
   if((direction > 0 && bid >= target) || (direction < 0 && ask <= target))
     {
      CloseAllOurPositions();
      g_grid_levels = 0;
      g_grid_direction = 0;
      return;
     }

   const double adverse_points = SafeDiv(-direction * (price - extreme), _Point);
   if(adverse_points >= {IN_grid_step_points} && count < {IN_max_levels})
     {
      double lots = InpLots * MathPow({IN_lot_multiplier}, count);
      lots = NormalizeLots(lots);
      const ENUM_ORDER_TYPE type = (direction > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
      if(SendMarketOrder(type, lots, 0.0, 0.0, "EAF_G" + IntegerToString(count)))
        {
         g_grid_levels = count + 1;
         g_grid_extreme = price;
         g_grid_direction = direction;
        }
     }
  }
