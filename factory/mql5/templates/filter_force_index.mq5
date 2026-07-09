//@SECTION INPUTS
input group "=== Filter {I}: Force Index ==="
input int {IN_force_period} = {P_force_period}; // Force Index smoothing period
//@SECTION GLOBALS
int g_f{I}_force_handle = INVALID_HANDLE;
//@SECTION INIT
//@SECTION RELEASE
   if(g_f{I}_force_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_force_handle);
//@SECTION LONG_EXPR
Filter{I}_Long()
//@SECTION SHORT_EXPR
Filter{I}_Short()
//@SECTION FUNCTIONS
bool Filter{I}_Ensure()
  {
   if(g_f{I}_force_handle != INVALID_HANDLE)
      return(true);
   g_f{I}_force_handle = iForce(_Symbol, _Period, {IN_force_period},
                                MODE_EMA, VOLUME_TICK);
   return(g_f{I}_force_handle != INVALID_HANDLE);
  }

bool Filter{I}_Long()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double fi[];
   if(!SafeCopyBuffer(g_f{I}_force_handle, 0, 1, 2, fi))
      return(false);
   return(fi[0] > 0.0 && fi[1] <= 0.0);
  }

bool Filter{I}_Short()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double fi[];
   if(!SafeCopyBuffer(g_f{I}_force_handle, 0, 1, 2, fi))
      return(false);
   return(fi[0] < 0.0 && fi[1] >= 0.0);
  }
