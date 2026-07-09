//@SECTION INPUTS
input group "=== Filter {I}: Liquidity Zone ==="
input int    {IN_zone_lookback} = {P_zone_lookback}; // Zone lookback (bars)
input double {IN_zone_points}   = {P_zone_points};   // Zone proximity (points)
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
   double lows[];
   double closes[];
   if(!SafeCopyLow(2, {IN_zone_lookback}, lows))
      return(false);
   if(!SafeCopyClose(1, 1, closes))
      return(false);
   const int idx = ArrayMinimum(lows);
   if(idx < 0)
      return(false);
   return(MathAbs(closes[0] - lows[idx]) <= {IN_zone_points} * _Point);
  }

bool Filter{I}_Short()
  {
   double highs[];
   double closes[];
   if(!SafeCopyHigh(2, {IN_zone_lookback}, highs))
      return(false);
   if(!SafeCopyClose(1, 1, closes))
      return(false);
   const int idx = ArrayMaximum(highs);
   if(idx < 0)
      return(false);
   return(MathAbs(highs[idx] - closes[0]) <= {IN_zone_points} * _Point);
  }
