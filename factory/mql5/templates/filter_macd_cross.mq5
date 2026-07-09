//@SECTION INPUTS
input group "=== Filter {I}: MACD Cross ==="
input int {IN_fast_ema}      = {P_fast_ema};      // MACD fast EMA
input int {IN_slow_ema}      = {P_slow_ema};      // MACD slow EMA
input int {IN_signal_period} = {P_signal_period}; // MACD signal period
//@SECTION GLOBALS
int g_f{I}_macd_handle = INVALID_HANDLE;
//@SECTION INIT
//@SECTION RELEASE
   if(g_f{I}_macd_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_macd_handle);
//@SECTION LONG_EXPR
Filter{I}_Long()
//@SECTION SHORT_EXPR
Filter{I}_Short()
//@SECTION FUNCTIONS
bool Filter{I}_Ensure()
  {
   if(g_f{I}_macd_handle != INVALID_HANDLE)
      return(true);
   g_f{I}_macd_handle = iMACD(_Symbol, _Period, {IN_fast_ema}, {IN_slow_ema},
                              {IN_signal_period}, PRICE_CLOSE);
   return(g_f{I}_macd_handle != INVALID_HANDLE);
  }

bool Filter{I}_Cross(bool &up, bool &down)
  {
   if(!Filter{I}_Ensure())
      return(false);
   double main[];
   double sig[];
   if(!SafeCopyBuffer(g_f{I}_macd_handle, 0, 1, 2, main))
      return(false);
   if(!SafeCopyBuffer(g_f{I}_macd_handle, 1, 1, 2, sig))
      return(false);
   up   = (main[0] > sig[0] && main[1] <= sig[1]);
   down = (main[0] < sig[0] && main[1] >= sig[1]);
   return(true);
  }

bool Filter{I}_Long()
  {
   bool up = false, down = false;
   if(!Filter{I}_Cross(up, down))
      return(false);
   return(up);
  }

bool Filter{I}_Short()
  {
   bool up = false, down = false;
   if(!Filter{I}_Cross(up, down))
      return(false);
   return(down);
  }
