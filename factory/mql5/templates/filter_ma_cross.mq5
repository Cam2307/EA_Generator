//@SECTION INPUTS
input group "=== Filter {I}: MA Cross ==="
input int {IN_fast_period} = {P_fast_period}; // Fast SMA period
input int {IN_slow_period} = {P_slow_period}; // Slow SMA period
//@SECTION GLOBALS
int g_f{I}_fast_handle = INVALID_HANDLE;
int g_f{I}_slow_handle = INVALID_HANDLE;
//@SECTION INIT
//@SECTION RELEASE
   if(g_f{I}_fast_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_fast_handle);
   if(g_f{I}_slow_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_slow_handle);
//@SECTION LONG_EXPR
Filter{I}_Long()
//@SECTION SHORT_EXPR
Filter{I}_Short()
//@SECTION FUNCTIONS
bool Filter{I}_Ensure()
  {
   if(g_f{I}_fast_handle == INVALID_HANDLE)
      g_f{I}_fast_handle = iMA(_Symbol, _Period, {IN_fast_period}, 0, MODE_SMA, PRICE_CLOSE);
   if(g_f{I}_slow_handle == INVALID_HANDLE)
      g_f{I}_slow_handle = iMA(_Symbol, _Period, {IN_slow_period}, 0, MODE_SMA, PRICE_CLOSE);
   return(g_f{I}_fast_handle != INVALID_HANDLE &&
          g_f{I}_slow_handle != INVALID_HANDLE);
  }

bool Filter{I}_Cross(bool &up, bool &down)
  {
   if(!Filter{I}_Ensure())
      return(false);
   double fast[];
   double slow[];
   if(!SafeCopyBuffer(g_f{I}_fast_handle, 0, 1, 2, fast))
      return(false);
   if(!SafeCopyBuffer(g_f{I}_slow_handle, 0, 1, 2, slow))
      return(false);
   up   = (fast[0] > slow[0] && fast[1] <= slow[1]);
   down = (fast[0] < slow[0] && fast[1] >= slow[1]);
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
