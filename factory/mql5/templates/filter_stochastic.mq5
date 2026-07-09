//@SECTION INPUTS
input group "=== Filter {I}: Stochastic ==="
input int    {IN_k_period}   = {P_k_period};   // Stochastic %K period
input double {IN_oversold}   = {P_oversold};   // Oversold threshold
input double {IN_overbought} = {P_overbought}; // Overbought threshold
//@SECTION GLOBALS
int g_f{I}_stoch_handle = INVALID_HANDLE;
//@SECTION INIT
//@SECTION RELEASE
   if(g_f{I}_stoch_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_stoch_handle);
//@SECTION LONG_EXPR
Filter{I}_Long()
//@SECTION SHORT_EXPR
Filter{I}_Short()
//@SECTION FUNCTIONS
bool Filter{I}_Ensure()
  {
   if(g_f{I}_stoch_handle != INVALID_HANDLE)
      return(true);
   g_f{I}_stoch_handle = iStochastic(_Symbol, _Period, {IN_k_period}, 3, 3,
                                     MODE_SMA, STO_LOWHIGH);
   return(g_f{I}_stoch_handle != INVALID_HANDLE);
  }

bool Filter{I}_Long()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double k[];
   if(!SafeCopyBuffer(g_f{I}_stoch_handle, 0, 1, 1, k))
      return(false);
   return(k[0] < {IN_oversold});
  }

bool Filter{I}_Short()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double k[];
   if(!SafeCopyBuffer(g_f{I}_stoch_handle, 0, 1, 1, k))
      return(false);
   return(k[0] > {IN_overbought});
  }
