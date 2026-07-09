//@SECTION INPUTS
input group "=== Filter {I}: Envelopes ==="
input int    {IN_env_period}    = {P_env_period};    // Envelope MA period
input double {IN_env_deviation} = {P_env_deviation}; // Envelope deviation (%)
//@SECTION GLOBALS
int g_f{I}_env_handle = INVALID_HANDLE;
//@SECTION INIT
//@SECTION RELEASE
   if(g_f{I}_env_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_env_handle);
//@SECTION LONG_EXPR
Filter{I}_Long()
//@SECTION SHORT_EXPR
Filter{I}_Short()
//@SECTION FUNCTIONS
bool Filter{I}_Ensure()
  {
   if(g_f{I}_env_handle != INVALID_HANDLE)
      return(true);
   g_f{I}_env_handle = iEnvelopes(_Symbol, _Period, {IN_env_period}, 0,
                                  MODE_SMA, PRICE_CLOSE, {IN_env_deviation});
   return(g_f{I}_env_handle != INVALID_HANDLE);
  }

bool Filter{I}_Long()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double lower[];
   double closes[];
   if(!SafeCopyBuffer(g_f{I}_env_handle, 1, 1, 1, lower))
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
   if(!SafeCopyBuffer(g_f{I}_env_handle, 0, 1, 1, upper))
      return(false);
   if(!SafeCopyClose(1, 1, closes))
      return(false);
   return(closes[0] > upper[0]);
  }
