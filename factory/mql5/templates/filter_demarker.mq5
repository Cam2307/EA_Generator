//@SECTION INPUTS
input group "=== Filter {I}: DeMarker ==="
input int    {IN_dem_period}     = {P_dem_period};     // DeMarker period
input double {IN_dem_oversold}   = {P_dem_oversold};   // Oversold threshold
input double {IN_dem_overbought} = {P_dem_overbought}; // Overbought threshold
//@SECTION GLOBALS
int g_f{I}_dem_handle = INVALID_HANDLE;
//@SECTION INIT
//@SECTION RELEASE
   if(g_f{I}_dem_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_dem_handle);
//@SECTION LONG_EXPR
Filter{I}_Long()
//@SECTION SHORT_EXPR
Filter{I}_Short()
//@SECTION FUNCTIONS
bool Filter{I}_Ensure()
  {
   if(g_f{I}_dem_handle != INVALID_HANDLE)
      return(true);
   g_f{I}_dem_handle = iDeMarker(_Symbol, _Period, {IN_dem_period});
   return(g_f{I}_dem_handle != INVALID_HANDLE);
  }

bool Filter{I}_Long()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double dem[];
   if(!SafeCopyBuffer(g_f{I}_dem_handle, 0, 1, 1, dem))
      return(false);
   return(dem[0] < {IN_dem_oversold});
  }

bool Filter{I}_Short()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double dem[];
   if(!SafeCopyBuffer(g_f{I}_dem_handle, 0, 1, 1, dem))
      return(false);
   return(dem[0] > {IN_dem_overbought});
  }
