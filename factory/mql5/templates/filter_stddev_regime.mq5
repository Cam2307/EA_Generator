//@SECTION INPUTS
input group "=== Filter {I}: StdDev Regime ==="
input int    {IN_std_period} = {P_std_period}; // Standard-deviation period
input double {IN_std_mult}   = {P_std_mult};   // Volatility-expansion multiple
//@SECTION GLOBALS
int g_f{I}_std_handle = INVALID_HANDLE;
//@SECTION INIT
//@SECTION RELEASE
   if(g_f{I}_std_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_std_handle);
//@SECTION LONG_EXPR
Filter{I}_Regime()
//@SECTION SHORT_EXPR
Filter{I}_Regime()
//@SECTION FUNCTIONS
bool Filter{I}_Ensure()
  {
   if(g_f{I}_std_handle != INVALID_HANDLE)
      return(true);
   g_f{I}_std_handle = iStdDev(_Symbol, _Period, {IN_std_period}, 0,
                               MODE_SMA, PRICE_CLOSE);
   return(g_f{I}_std_handle != INVALID_HANDLE);
  }

bool Filter{I}_Regime()
  {
   if(!Filter{I}_Ensure())
      return(false);
   const int period = (int){IN_std_period};
   if(period < 1)
      return(false);
   double std[];
   if(!SafeCopyBuffer(g_f{I}_std_handle, 0, 1, period + 1, std))
      return(false);
   double sum = 0.0;
   for(int i = 1; i <= period; i++)
      sum += std[i];
   const double avg = SafeDiv(sum, (double)period);
   if(avg <= 0.0)
      return(false);
   return(std[0] > {IN_std_mult} * avg);
  }
