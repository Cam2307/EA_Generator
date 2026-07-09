//@SECTION INPUTS
input group "=== Filter {I}: Ichimoku ==="
input int {IN_tenkan} = {P_tenkan}; // Tenkan-sen period
input int {IN_kijun}  = {P_kijun};  // Kijun-sen period
input int {IN_senkou} = {P_senkou}; // Senkou Span B period
//@SECTION GLOBALS
int g_f{I}_ichi_handle = INVALID_HANDLE;
//@SECTION INIT
//@SECTION RELEASE
   if(g_f{I}_ichi_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_ichi_handle);
//@SECTION LONG_EXPR
Filter{I}_Long()
//@SECTION SHORT_EXPR
Filter{I}_Short()
//@SECTION FUNCTIONS
bool Filter{I}_Ensure()
  {
   if(g_f{I}_ichi_handle != INVALID_HANDLE)
      return(true);
   g_f{I}_ichi_handle = iIchimoku(_Symbol, _Period, {IN_tenkan}, {IN_kijun},
                                  {IN_senkou});
   return(g_f{I}_ichi_handle != INVALID_HANDLE);
  }

bool Filter{I}_Cross(bool &up, bool &down)
  {
   if(!Filter{I}_Ensure())
      return(false);
   double tenkan[];
   double kijun[];
   if(!SafeCopyBuffer(g_f{I}_ichi_handle, 0, 1, 2, tenkan))
      return(false);
   if(!SafeCopyBuffer(g_f{I}_ichi_handle, 1, 1, 2, kijun))
      return(false);
   up   = (tenkan[0] > kijun[0] && tenkan[1] <= kijun[1]);
   down = (tenkan[0] < kijun[0] && tenkan[1] >= kijun[1]);
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
