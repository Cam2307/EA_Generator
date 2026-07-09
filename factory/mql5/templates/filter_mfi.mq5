//@SECTION INPUTS
input group "=== Filter {I}: Money Flow Index ==="
input int    {IN_mfi_period}     = {P_mfi_period};     // MFI period
input double {IN_mfi_oversold}   = {P_mfi_oversold};   // Oversold threshold
input double {IN_mfi_overbought} = {P_mfi_overbought}; // Overbought threshold
//@SECTION GLOBALS
int g_f{I}_mfi_handle = INVALID_HANDLE;
//@SECTION INIT
//@SECTION RELEASE
   if(g_f{I}_mfi_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_mfi_handle);
//@SECTION LONG_EXPR
Filter{I}_Long()
//@SECTION SHORT_EXPR
Filter{I}_Short()
//@SECTION FUNCTIONS
bool Filter{I}_Ensure()
  {
   if(g_f{I}_mfi_handle != INVALID_HANDLE)
      return(true);
   g_f{I}_mfi_handle = iMFI(_Symbol, _Period, {IN_mfi_period}, VOLUME_TICK);
   return(g_f{I}_mfi_handle != INVALID_HANDLE);
  }

bool Filter{I}_Long()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double mfi[];
   if(!SafeCopyBuffer(g_f{I}_mfi_handle, 0, 1, 1, mfi))
      return(false);
   return(mfi[0] < {IN_mfi_oversold});
  }

bool Filter{I}_Short()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double mfi[];
   if(!SafeCopyBuffer(g_f{I}_mfi_handle, 0, 1, 1, mfi))
      return(false);
   return(mfi[0] > {IN_mfi_overbought});
  }
