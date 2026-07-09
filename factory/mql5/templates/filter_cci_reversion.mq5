//@SECTION INPUTS
input group "=== Filter {I}: CCI Reversion ==="
input int    {IN_cci_period} = {P_cci_period}; // CCI period
input double {IN_cci_level}  = {P_cci_level};  // CCI reversion level
//@SECTION GLOBALS
int g_f{I}_cci_handle = INVALID_HANDLE;
//@SECTION INIT
//@SECTION RELEASE
   if(g_f{I}_cci_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_cci_handle);
//@SECTION LONG_EXPR
Filter{I}_Long()
//@SECTION SHORT_EXPR
Filter{I}_Short()
//@SECTION FUNCTIONS
bool Filter{I}_Ensure()
  {
   if(g_f{I}_cci_handle != INVALID_HANDLE)
      return(true);
   g_f{I}_cci_handle = iCCI(_Symbol, _Period, {IN_cci_period}, PRICE_TYPICAL);
   return(g_f{I}_cci_handle != INVALID_HANDLE);
  }

bool Filter{I}_Long()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double cci[];
   if(!SafeCopyBuffer(g_f{I}_cci_handle, 0, 1, 1, cci))
      return(false);
   return(cci[0] < -{IN_cci_level});
  }

bool Filter{I}_Short()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double cci[];
   if(!SafeCopyBuffer(g_f{I}_cci_handle, 0, 1, 1, cci))
      return(false);
   return(cci[0] > {IN_cci_level});
  }
