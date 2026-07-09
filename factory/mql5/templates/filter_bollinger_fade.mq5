//@SECTION INPUTS
input group "=== Filter {I}: Bollinger Fade ==="
input int    {IN_bb_period} = {P_bb_period}; // Bollinger period
input double {IN_bb_dev}    = {P_bb_dev};    // Bollinger deviation
//@SECTION GLOBALS
int g_f{I}_bb_handle = INVALID_HANDLE;
//@SECTION INIT
//@SECTION RELEASE
   if(g_f{I}_bb_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_bb_handle);
//@SECTION LONG_EXPR
Filter{I}_Long()
//@SECTION SHORT_EXPR
Filter{I}_Short()
//@SECTION FUNCTIONS
bool Filter{I}_Ensure()
  {
   if(g_f{I}_bb_handle != INVALID_HANDLE)
      return(true);
   g_f{I}_bb_handle = iBands(_Symbol, _Period, {IN_bb_period}, 0, {IN_bb_dev}, PRICE_CLOSE);
   return(g_f{I}_bb_handle != INVALID_HANDLE);
  }

bool Filter{I}_Long()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double lower[];
   double closes[];
   if(!SafeCopyBuffer(g_f{I}_bb_handle, 2, 1, 1, lower))
      return(false);
   if(!SafeCopyClose(1, 1, closes))
      return(false);
   return(closes[0] < lower[0]);
  }

bool Filter{I}_Short()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double upper[];
   double closes[];
   if(!SafeCopyBuffer(g_f{I}_bb_handle, 1, 1, 1, upper))
      return(false);
   if(!SafeCopyClose(1, 1, closes))
      return(false);
   return(closes[0] > upper[0]);
  }
