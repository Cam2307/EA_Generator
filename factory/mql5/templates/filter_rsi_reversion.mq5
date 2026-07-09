//@SECTION INPUTS
input group "=== Filter {I}: RSI Reversion ==="
input int    {IN_rsi_period} = {P_rsi_period}; // RSI period
input double {IN_oversold}   = {P_oversold};   // Oversold threshold
input double {IN_overbought} = {P_overbought}; // Overbought threshold
//@SECTION GLOBALS
int g_f{I}_rsi_handle = INVALID_HANDLE;
//@SECTION INIT
//@SECTION RELEASE
   if(g_f{I}_rsi_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_rsi_handle);
//@SECTION LONG_EXPR
Filter{I}_Long()
//@SECTION SHORT_EXPR
Filter{I}_Short()
//@SECTION FUNCTIONS
bool Filter{I}_Ensure()
  {
   if(g_f{I}_rsi_handle != INVALID_HANDLE)
      return(true);
   g_f{I}_rsi_handle = iRSI(_Symbol, _Period, {IN_rsi_period}, PRICE_CLOSE);
   return(g_f{I}_rsi_handle != INVALID_HANDLE);
  }

bool Filter{I}_Long()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double rsi[];
   if(!SafeCopyBuffer(g_f{I}_rsi_handle, 0, 1, 1, rsi))
      return(false);
   return(rsi[0] < {IN_oversold});
  }

bool Filter{I}_Short()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double rsi[];
   if(!SafeCopyBuffer(g_f{I}_rsi_handle, 0, 1, 1, rsi))
      return(false);
   return(rsi[0] > {IN_overbought});
  }
