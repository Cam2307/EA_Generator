//@SECTION INPUTS
input group "=== Execution: Standard SL/TP ==="
input double {IN_sl_points} = {P_sl_points}; // Stop loss (points)
input double {IN_tp_points} = {P_tp_points}; // Take profit (points)
//@SECTION FUNCTIONS
void OpenEntry(const ENUM_ORDER_TYPE type)
  {
   const double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   const double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   const double sl_pts = TM_InitialStopPoints({IN_sl_points});
   const double tp_pts = TM_TakeProfitPoints(sl_pts, {IN_tp_points});
   const double lots   = TM_Lots(sl_pts);
   double sl = 0.0, tp = 0.0;
   if(type == ORDER_TYPE_BUY)
     {
      sl = (sl_pts > 0.0) ? ask - sl_pts * _Point : 0.0;
      tp = (tp_pts > 0.0) ? ask + tp_pts * _Point : 0.0;
     }
   else
     {
      sl = (sl_pts > 0.0) ? bid + sl_pts * _Point : 0.0;
      tp = (tp_pts > 0.0) ? bid - tp_pts * _Point : 0.0;
     }
   if(SendMarketOrder(type, lots, sl, tp, "EAF_STD"))
      g_tm_trades_today++;
  }

void ManagePositions()
  {
   // standard SL/TP: broker-side exits only, nothing to manage
  }
