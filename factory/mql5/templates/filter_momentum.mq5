//@SECTION INPUTS
input group "=== Filter {I}: Momentum ==="
input int    {IN_mom_period}    = {P_mom_period};    // Momentum period
input double {IN_mom_threshold} = {P_mom_threshold}; // Momentum threshold (around 100)
//@SECTION GLOBALS
int g_f{I}_mom_handle = INVALID_HANDLE;
//@SECTION INIT
//@SECTION RELEASE
   if(g_f{I}_mom_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_mom_handle);
//@SECTION LONG_EXPR
Filter{I}_Long()
//@SECTION SHORT_EXPR
Filter{I}_Short()
//@SECTION FUNCTIONS
bool Filter{I}_Ensure()
  {
   if(g_f{I}_mom_handle != INVALID_HANDLE)
      return(true);
   g_f{I}_mom_handle = iMomentum(_Symbol, _Period, {IN_mom_period}, PRICE_CLOSE);
   return(g_f{I}_mom_handle != INVALID_HANDLE);
  }

bool Filter{I}_Long()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double mom[];
   if(!SafeCopyBuffer(g_f{I}_mom_handle, 0, 1, 1, mom))
      return(false);
   return(mom[0] > 100.0 + {IN_mom_threshold});
  }

bool Filter{I}_Short()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double mom[];
   if(!SafeCopyBuffer(g_f{I}_mom_handle, 0, 1, 1, mom))
      return(false);
   return(mom[0] < 100.0 - {IN_mom_threshold});
  }
