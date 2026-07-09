//@SECTION INPUTS
input group "=== Filter {I}: Volatility Regime ==="
input int    {IN_atr_period}   = {P_atr_period};   // ATR period
input double {IN_atr_mult_min} = {P_atr_mult_min}; // Min ATR vs baseline multiple
//@SECTION GLOBALS
int g_f{I}_atr_handle = INVALID_HANDLE;
int g_f{I}_ma_handle  = INVALID_HANDLE;
//@SECTION INIT
//@SECTION RELEASE
   if(g_f{I}_atr_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_atr_handle);
   if(g_f{I}_ma_handle != INVALID_HANDLE)
      IndicatorRelease(g_f{I}_ma_handle);
//@SECTION LONG_EXPR
Filter{I}_Long()
//@SECTION SHORT_EXPR
Filter{I}_Short()
//@SECTION FUNCTIONS
bool Filter{I}_Ensure()
  {
   if(g_f{I}_atr_handle == INVALID_HANDLE)
      g_f{I}_atr_handle = iATR(_Symbol, _Period, {IN_atr_period});
   if(g_f{I}_ma_handle == INVALID_HANDLE)
      g_f{I}_ma_handle = iMA(_Symbol, _Period, {IN_atr_period} * 2, 0, MODE_SMA, PRICE_CLOSE);
   return(g_f{I}_atr_handle != INVALID_HANDLE &&
          g_f{I}_ma_handle != INVALID_HANDLE);
  }

bool Filter{I}_Gate(double &last_close, double &last_ma)
  {
   if(!Filter{I}_Ensure())
      return(false);
   double atr[];
   double ma[];
   double closes[];
   const int baseline_bars = 100;
   if(!SafeCopyBuffer(g_f{I}_atr_handle, 0, 1, baseline_bars, atr))
      return(false);
   if(!SafeCopyBuffer(g_f{I}_ma_handle, 0, 1, 1, ma))
      return(false);
   if(!SafeCopyClose(1, 1, closes))
      return(false);
   double sum = 0.0;
   for(int i = 0; i < baseline_bars; i++)
      sum += atr[i];
   const double baseline = SafeDiv(sum, (double)baseline_bars);
   if(baseline <= 0.0)
      return(false);
   last_close = closes[0];
   last_ma    = ma[0];
   return(atr[0] > {IN_atr_mult_min} * baseline);
  }

bool Filter{I}_Long()
  {
   double c = 0.0, m = 0.0;
   if(!Filter{I}_Gate(c, m))
      return(false);
   return(c > m);
  }

bool Filter{I}_Short()
  {
   double c = 0.0, m = 0.0;
   if(!Filter{I}_Gate(c, m))
      return(false);
   return(c < m);
  }
