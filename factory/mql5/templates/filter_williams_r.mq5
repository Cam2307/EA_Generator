//@SECTION INPUTS
input group "=== Filter {I}: Williams %R ==="
input int    {IN_wpr_period}     = {P_wpr_period};     // Williams %R period
input double {IN_wpr_oversold}   = {P_wpr_oversold};   // Oversold level (-100..0)
input double {IN_wpr_overbought} = {P_wpr_overbought}; // Overbought level (-100..0)
//@SECTION GLOBALS
int g_f{I}_wpr_handle = INVALID_HANDLE;
//@SECTION INIT
//@SECTION RELEASE
   if(g_f{I}_wpr_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_wpr_handle);
//@SECTION LONG_EXPR
Filter{I}_Long()
//@SECTION SHORT_EXPR
Filter{I}_Short()
//@SECTION FUNCTIONS
bool Filter{I}_Ensure()
  {
   if(g_f{I}_wpr_handle != INVALID_HANDLE)
      return(true);
   g_f{I}_wpr_handle = iWPR(_Symbol, _Period, {IN_wpr_period});
   return(g_f{I}_wpr_handle != INVALID_HANDLE);
  }

bool Filter{I}_Long()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double wpr[];
   if(!SafeCopyBuffer(g_f{I}_wpr_handle, 0, 1, 1, wpr))
      return(false);
   return(wpr[0] < {IN_wpr_oversold});
  }

bool Filter{I}_Short()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double wpr[];
   if(!SafeCopyBuffer(g_f{I}_wpr_handle, 0, 1, 1, wpr))
      return(false);
   return(wpr[0] > {IN_wpr_overbought});
  }
