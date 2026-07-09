//@SECTION INPUTS
input group "=== Filter {I}: Volume Surge ==="
input int    {IN_vol_period} = {P_vol_period}; // Volume average period (bars)
input double {IN_vol_mult}   = {P_vol_mult};   // Surge multiple of average volume
//@SECTION GLOBALS
int g_f{I}_vol_handle = INVALID_HANDLE;
//@SECTION INIT
//@SECTION RELEASE
   if(g_f{I}_vol_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_vol_handle);
//@SECTION LONG_EXPR
Filter{I}_Surge()
//@SECTION SHORT_EXPR
Filter{I}_Surge()
//@SECTION FUNCTIONS
bool Filter{I}_Ensure()
  {
   if(g_f{I}_vol_handle != INVALID_HANDLE)
      return(true);
   g_f{I}_vol_handle = iVolumes(_Symbol, _Period, VOLUME_TICK);
   return(g_f{I}_vol_handle != INVALID_HANDLE);
  }

bool Filter{I}_Surge()
  {
   if(!Filter{I}_Ensure())
      return(false);
   const int period = (int){IN_vol_period};
   if(period < 1)
      return(false);
   double vol[];
   if(!SafeCopyBuffer(g_f{I}_vol_handle, 0, 1, period + 1, vol))
      return(false);
   double sum = 0.0;
   for(int i = 1; i <= period; i++)
      sum += vol[i];
   const double avg = SafeDiv(sum, (double)period);
   if(avg <= 0.0)
      return(false);
   return(vol[0] > {IN_vol_mult} * avg);
  }
