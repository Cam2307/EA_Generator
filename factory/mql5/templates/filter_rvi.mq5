//@SECTION INPUTS
input group "=== Filter {I}: Relative Vigor Index ==="
input int {IN_rvi_period} = {P_rvi_period}; // RVI period
//@SECTION GLOBALS
int g_f{I}_rvi_handle = INVALID_HANDLE;
//@SECTION INIT
//@SECTION RELEASE
   if(g_f{I}_rvi_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_rvi_handle);
//@SECTION LONG_EXPR
Filter{I}_Long()
//@SECTION SHORT_EXPR
Filter{I}_Short()
//@SECTION FUNCTIONS
bool Filter{I}_Ensure()
  {
   if(g_f{I}_rvi_handle != INVALID_HANDLE)
      return(true);
   g_f{I}_rvi_handle = iRVI(_Symbol, _Period, {IN_rvi_period});
   return(g_f{I}_rvi_handle != INVALID_HANDLE);
  }

bool Filter{I}_Cross(bool &up, bool &down)
  {
   if(!Filter{I}_Ensure())
      return(false);
   double main[];
   double sig[];
   if(!SafeCopyBuffer(g_f{I}_rvi_handle, 0, 1, 2, main))
      return(false);
   if(!SafeCopyBuffer(g_f{I}_rvi_handle, 1, 1, 2, sig))
      return(false);
   up   = (main[0] > sig[0] && main[1] <= sig[1]);
   down = (main[0] < sig[0] && main[1] >= sig[1]);
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
