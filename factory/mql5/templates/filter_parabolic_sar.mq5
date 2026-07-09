//@SECTION INPUTS
input group "=== Filter {I}: Parabolic SAR ==="
input double {IN_sar_step} = {P_sar_step}; // SAR acceleration step
input double {IN_sar_max}  = {P_sar_max};  // SAR maximum acceleration
//@SECTION GLOBALS
int g_f{I}_sar_handle = INVALID_HANDLE;
//@SECTION INIT
//@SECTION RELEASE
   if(g_f{I}_sar_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_sar_handle);
//@SECTION LONG_EXPR
Filter{I}_Long()
//@SECTION SHORT_EXPR
Filter{I}_Short()
//@SECTION FUNCTIONS
bool Filter{I}_Ensure()
  {
   if(g_f{I}_sar_handle != INVALID_HANDLE)
      return(true);
   g_f{I}_sar_handle = iSAR(_Symbol, _Period, {IN_sar_step}, {IN_sar_max});
   return(g_f{I}_sar_handle != INVALID_HANDLE);
  }

bool Filter{I}_Long()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double sar[];
   double closes[];
   if(!SafeCopyBuffer(g_f{I}_sar_handle, 0, 1, 1, sar))
      return(false);
   if(!SafeCopyClose(1, 1, closes))
      return(false);
   return(closes[0] > sar[0]);
  }

bool Filter{I}_Short()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double sar[];
   double closes[];
   if(!SafeCopyBuffer(g_f{I}_sar_handle, 0, 1, 1, sar))
      return(false);
   if(!SafeCopyClose(1, 1, closes))
      return(false);
   return(closes[0] < sar[0]);
  }
