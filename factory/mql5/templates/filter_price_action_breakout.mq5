//@SECTION INPUTS
input group "=== Filter {I}: Price Action Breakout ==="
input int    {IN_lookback}      = {P_lookback};      // Breakout lookback (bars)
input double {IN_buffer_points} = {P_buffer_points}; // Breakout buffer (points)
//@SECTION GLOBALS
//@SECTION INIT
//@SECTION RELEASE
//@SECTION LONG_EXPR
Filter{I}_Long()
//@SECTION SHORT_EXPR
Filter{I}_Short()
//@SECTION FUNCTIONS
bool Filter{I}_Long()
  {
   double highs[];
   double closes[];
   if(!SafeCopyHigh(2, {IN_lookback}, highs))
      return(false);
   if(!SafeCopyClose(1, 1, closes))
      return(false);
   const int idx = ArrayMaximum(highs);
   if(idx < 0)
      return(false);
   return(closes[0] > highs[idx] + {IN_buffer_points} * _Point);
  }

bool Filter{I}_Short()
  {
   double lows[];
   double closes[];
   if(!SafeCopyLow(2, {IN_lookback}, lows))
      return(false);
   if(!SafeCopyClose(1, 1, closes))
      return(false);
   const int idx = ArrayMinimum(lows);
   if(idx < 0)
      return(false);
   return(closes[0] < lows[idx] - {IN_buffer_points} * _Point);
  }
