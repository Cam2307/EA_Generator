//@SECTION INPUTS
input group "=== Execution: Partial Close ==="
input double {IN_sl_points}         = {P_sl_points};         // Stop loss (points)
input double {IN_tp_points}         = {P_tp_points};         // Take profit (points)
input double {IN_partial_tp_points} = {P_partial_tp_points}; // Partial-close trigger (points)
input double {IN_partial_fraction}  = {P_partial_fraction};  // Fraction closed at trigger
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
   if(SendMarketOrder(type, lots, sl, tp, "EAF_PC"))
      g_tm_trades_today++;
  }

void ManagePositions()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      const ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket) || !IsOurPosition())
         continue;

      const int dir = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 1 : -1;
      const ENUM_ORDER_TYPE ptype = (dir > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
      const double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      const double sl    = PositionGetDouble(POSITION_SL);
      const double vol   = PositionGetDouble(POSITION_VOLUME);

      if(MathAbs(sl - entry) < _Point)
         continue;

      const double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      const double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      const double price = (dir > 0) ? bid : ask;
      const double gain_points = SafeDiv(dir * (price - entry), _Point);
      if(gain_points < {IN_partial_tp_points})
         continue;

      const double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
      const double lot_min  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
      double part = vol * {IN_partial_fraction};
      if(lot_step > 0.0)
         part = MathFloor(SafeDiv(part, lot_step)) * lot_step;
      if(part >= lot_min && (vol - part) >= lot_min)
        {
         if(!SafeClosePosition(ticket, part))
            continue;
        }
      if(PositionSelectByTicket(ticket))
        {
         const double tp = PositionGetDouble(POSITION_TP);
         if(!FreezeOK(ptype, entry, entry, tp))
            continue;
         for(int attempt = 1; attempt <= InpMaxRetries; attempt++)
           {
            if(g_trade.PositionModify(ticket, NormalizeDouble(entry, _Digits), tp) &&
               RetcodeOK(g_trade.ResultRetcode()))
               break;
            if(!RetcodeTransient(g_trade.ResultRetcode()))
               break;
            Sleep(InpRetryBaseMs * attempt);
           }
        }
     }
  }
