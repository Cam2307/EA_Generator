//@SECTION INPUTS
input group "=== Filter {I}: ADX Trend ==="
input int    {IN_adx_period} = {P_adx_period}; // ADX period
input double {IN_adx_min}    = {P_adx_min};    // Minimum ADX to trade
//@SECTION GLOBALS
int g_f{I}_adx_handle = INVALID_HANDLE;
//@SECTION INIT
//@SECTION RELEASE
   if(g_f{I}_adx_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_adx_handle);
//@SECTION LONG_EXPR
Filter{I}_Long()
//@SECTION SHORT_EXPR
Filter{I}_Short()
//@SECTION FUNCTIONS
bool Filter{I}_Ensure()
  {
   if(g_f{I}_adx_handle != INVALID_HANDLE)
      return(true);
   g_f{I}_adx_handle = iADX(_Symbol, _Period, {IN_adx_period});
   return(g_f{I}_adx_handle != INVALID_HANDLE);
  }

bool Filter{I}_Read(double &adx, double &plus_di, double &minus_di)
  {
   if(!Filter{I}_Ensure())
      return(false);
   double a[];
   double p[];
   double m[];
   if(!SafeCopyBuffer(g_f{I}_adx_handle, 0, 1, 1, a))
      return(false);
   if(!SafeCopyBuffer(g_f{I}_adx_handle, 1, 1, 1, p))
      return(false);
   if(!SafeCopyBuffer(g_f{I}_adx_handle, 2, 1, 1, m))
      return(false);
   adx = a[0];
   plus_di = p[0];
   minus_di = m[0];
   return(true);
  }

bool Filter{I}_Long()
  {
   double adx = 0.0, plus_di = 0.0, minus_di = 0.0;
   if(!Filter{I}_Read(adx, plus_di, minus_di))
      return(false);
   return(adx > {IN_adx_min} && plus_di > minus_di);
  }

bool Filter{I}_Short()
  {
   double adx = 0.0, plus_di = 0.0, minus_di = 0.0;
   if(!Filter{I}_Read(adx, plus_di, minus_di))
      return(false);
   return(adx > {IN_adx_min} && minus_di > plus_di);
  }
