//@SECTION INPUTS
input group "=== Filter {I}: DEMA Cross ==="
input int {IN_dema_fast} = {P_dema_fast}; // Fast DEMA period
input int {IN_dema_slow} = {P_dema_slow}; // Slow DEMA period
//@SECTION GLOBALS
int g_f{I}_dema_fast_handle = INVALID_HANDLE;
int g_f{I}_dema_slow_handle = INVALID_HANDLE;
//@SECTION INIT
//@SECTION RELEASE
   if(g_f{I}_dema_fast_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_dema_fast_handle);
   if(g_f{I}_dema_slow_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_dema_slow_handle);
//@SECTION LONG_EXPR
Filter{I}_Long()
//@SECTION SHORT_EXPR
Filter{I}_Short()
//@SECTION FUNCTIONS
bool Filter{I}_Ensure()
  {
   if(g_f{I}_dema_fast_handle == INVALID_HANDLE)
      g_f{I}_dema_fast_handle = iDEMA(_Symbol, _Period, {IN_dema_fast}, 0, PRICE_CLOSE);
   if(g_f{I}_dema_slow_handle == INVALID_HANDLE)
      g_f{I}_dema_slow_handle = iDEMA(_Symbol, _Period, {IN_dema_slow}, 0, PRICE_CLOSE);
   return(g_f{I}_dema_fast_handle != INVALID_HANDLE &&
          g_f{I}_dema_slow_handle != INVALID_HANDLE);
  }

bool Filter{I}_Cross(bool &up, bool &down)
  {
   if(!Filter{I}_Ensure())
      return(false);
   double fast[];
   double slow[];
   if(!SafeCopyBuffer(g_f{I}_dema_fast_handle, 0, 1, 2, fast))
      return(false);
   if(!SafeCopyBuffer(g_f{I}_dema_slow_handle, 0, 1, 2, slow))
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
