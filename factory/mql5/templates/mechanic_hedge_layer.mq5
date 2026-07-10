//@SECTION INPUTS
input group "=== Execution: Hedge Layer ==="
input double {IN_sl_points}            = {P_sl_points};            // Stop loss (points)
input double {IN_tp_points}            = {P_tp_points};            // Take profit (points)
input double {IN_hedge_trigger_points} = {P_hedge_trigger_points}; // Hedge trigger drawdown (points)
input double {IN_hedge_ratio}          = {P_hedge_ratio};          // Hedge lot ratio
//@SECTION GLOBALS
bool g_hedge_active = false;
//@SECTION FUNCTIONS
void OpenEntry(const ENUM_ORDER_TYPE type)
  {
   const double lots = NormalizeLots(InpLots * TM_RegimeLotMult());
   const double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   const double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl = 0.0, tp = 0.0;
   if(type == ORDER_TYPE_BUY)
     {
      sl = ask - {IN_sl_points} * _Point;
      tp = ask + {IN_tp_points} * _Point;
     }
   else
     {
      sl = bid + {IN_sl_points} * _Point;
      tp = bid - {IN_tp_points} * _Point;
     }
   if(SendMarketOrder(type, lots, sl, tp, "EAF_P"))
     {
      g_hedge_active = false;
      g_tm_trades_today++;
     }
  }

bool FindPrimaryTicket(ulong &ticket_out)
  {
   ticket_out = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      const ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket) || !IsOurPosition())
         continue;
      ticket_out = ticket;
      return(true);
     }
   return(false);
  }

void ManagePositions()
  {
   ulong primary = 0;
   if(!FindPrimaryTicket(primary))
     {
      g_hedge_active = false;
      return;
     }
   if(!PositionSelectByTicket(primary))
      return;

   const int dir = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 1 : -1;
   const double entry = PositionGetDouble(POSITION_PRICE_OPEN);
   const double vol   = PositionGetDouble(POSITION_VOLUME);
   const double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   const double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   const double price = (dir > 0) ? bid : ask;

   // netting: opposite order reduces/closes — use basket stop instead of hedge
   if(!IsHedgingAccount())
     {
      const double adverse_points = SafeDiv(-dir * (price - entry), _Point);
      if(adverse_points >= {IN_hedge_trigger_points} * 2.0)
         CloseAllOurPositions();
      return;
     }

   if(g_hedge_active)
     {
      if(BasketFloatingProfit() >= 0.0)
         CloseAllOurPositions();
      return;
     }

   const double adverse_points = SafeDiv(-dir * (price - entry), _Point);
   if(adverse_points >= {IN_hedge_trigger_points})
     {
      const double lots = NormalizeLots(vol * {IN_hedge_ratio});
      const ENUM_ORDER_TYPE type = (dir > 0) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
      if(SendMarketOrder(type, lots, 0.0, 0.0, "EAF_H"))
         g_hedge_active = true;
     }
  }
