//@SECTION INPUTS
input group "=== Filter {I}: Awesome Oscillator ==="
input double {IN_ao_threshold} = {P_ao_threshold}; // Zero-line threshold
//@SECTION GLOBALS
int g_f{I}_ao_handle = INVALID_HANDLE;
//@SECTION INIT
//@SECTION RELEASE
   if(g_f{I}_ao_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_ao_handle);
//@SECTION LONG_EXPR
Filter{I}_Long()
//@SECTION SHORT_EXPR
Filter{I}_Short()
//@SECTION FUNCTIONS
bool Filter{I}_Ensure()
  {
   if(g_f{I}_ao_handle != INVALID_HANDLE)
      return(true);
   g_f{I}_ao_handle = iAO(_Symbol, _Period);
   return(g_f{I}_ao_handle != INVALID_HANDLE);
  }

bool Filter{I}_Long()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double ao[];
   if(!SafeCopyBuffer(g_f{I}_ao_handle, 0, 1, 2, ao))
      return(false);
   return(ao[0] > {IN_ao_threshold} && ao[1] <= {IN_ao_threshold});
  }

bool Filter{I}_Short()
  {
   if(!Filter{I}_Ensure())
      return(false);
   double ao[];
   if(!SafeCopyBuffer(g_f{I}_ao_handle, 0, 1, 2, ao))
      return(false);
   return(ao[0] < -{IN_ao_threshold} && ao[1] >= -{IN_ao_threshold});
  }
