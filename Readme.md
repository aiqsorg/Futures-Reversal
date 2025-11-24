# Resistance Pattern Trading Strategy

## Overview

Broadly this strategy hinges on the idea that many traders engage in trading leveraged futures and use technical analysis and this creates a market phenomenon that can be exploited through the flaws of technical analysis. A trader will engage with what they see as a resistance pattern, and will wait for their idea to be validated and immediately short, but not get out quick enough or neglect that any new market participants will invalidate their pattern, and this will lead to a forced upward pattern as they may be forced to cover their leveraged short position.

## Essential Problems and Solutions

We have a few essential problems we need to really be able to tackle before we can really take advantage of this phenomenon. So first of all, in technical analysis, traders can just look at a chart to be able to see where maxima and minima exist, but depending on the timeframe you do it on, it can completely change how it looks. So the attempt to solve this is being able to reduce the time frames of these charts into specific component signals that represent multiple time frames.

### Multi-Timeframe Signal Composition

Take for instance a 5 minute timeframe which has twice the amount of data as the 10 minute signal. So then the 10 minute signal can be interpolated to be the same length as the 5 minute signal, and they can be averaged in different weights to put more emphasis on certain parts, with a 75/25 weighting reflecting more weight on the first signal of 5 minutes and less weight on the 10 minutes. But this creates its own signal that can additionally be carried forward and interpolated and averaged with other time frames and so on. Weights could be changed but this example uses the same weights along all relevant steps.

These signals have maxima that can be detected in though scipy, and the time periods in which these maxima occur on the composite signals can be projected back onto the original price data for actual price information, and the distance between the component maxima and minima can be integrated between to measure magnitude of jumps that is representative across time frames. These integrals can be normalized and classified to where a relatively big jump across other jumps.

Another point of confusion to clarify is that these extrema are time discrete and depending on which composite signals chosen or how they are constructed they may appear differently, and their spacing is independent of time, as they are simply projected maxima and minima from the composite signal.

### Normalization Considerations

However there is another circumstance to consider, which is the original price data also must be normalized because if not, an integral would capture more about the price levels at a particular time period than the pure shape of the jump. But in forward testing, normalization would require the whole prices be constantly adjusted based on the new prices. So it is reasonable thus that with a large amount of historical data, the forward testing could just use a linear regression to assign normalization values from the historicals (possibly a 1 year period) which could be updated at the end of a trading day on a rolling bases.

## Dual Signal Implementation

But back to the 75/25 composite that will give us an initial jump to view, and we can additionally use a 40/60 signal starting from 1 minute data with the same interpolation and averaging process as before to continue building off of and seeing if this resistance continues to hold after the initial jump has been established, which is marked not exceeding a particular decimal multiple of the original integral, and the reversal would be completed if on the same signal as the original integral, the new integral exceeded a particular decimal multiple of the original integral.

### Current Algorithm Structure

The implementation creates two different composite signals for different purposes. The first is a 40/60 composite from 1-minute timeframes for bounded resistance detection, where we merge 1min, 5min, 10min, 15min, and 30min data using the progressive weighted averaging approach. The second is a 50/50 composite from 5-minute timeframes for reversal detection, merging 5min, 10min, 15min, 30min, and 60min data.

The algorithm finds turning points in both composites using scipy peak detection, then calculates momentum integrals on the original 1-minute normalized data using the turning points from the 40/60 composite. We filter out extreme outliers with Z-scores greater than 15 to prevent skewing, then recalculate Z-scores on the filtered data to find statistically significant jumps.

For resistance pattern monitoring, we focus on positive significant jumps only. When a jump is detected with a Z-score above the threshold (currently 0.8), we monitor subsequent movements to see if they stay within bounds (0.6x the original integral magnitude) or violate the resistance. A reversal is confirmed when a movement exceeds the negative threshold (-1.0x the original integral magnitude).

## Current Implementation Status

For now this is the extent of the work, but later plan to pick up volatility monitoring from after the reversal through ATR, and seeing when it falls sharply, to establish the falling is done and with new buying in the market the reversal may be imminent and shorts may covered their positions in a buy cycle.

### Technical Implementation Details

The algorithm processes full year data from QuantConnect, creating multiple timeframes and generating the dual composite signals. It uses progressive sequential merging where each timeframe is merged with the growing composite using the specified weights. The turning point detection uses configurable prominence values, and momentum integrals are calculated using trapezoidal integration between consecutive turning points.

Statistical analysis is performed using Z-score normalization to identify significant movements, with filtering to remove extreme outliers that could skew the analysis. The resistance pattern monitoring tracks bounded movements, violations, and reversals, providing comprehensive analysis of each significant jump's behavior.

The visualization system creates comprehensive plots showing the original data versus composites, momentum integral distributions, Z-score evolution over time, pattern completion summaries, and individual resistance pattern behaviors. This provides detailed insight into how the algorithm identifies and tracks resistance patterns across the entire dataset.

## Future Development

The next phase will incorporate volatility analysis through Average True Range (ATR) monitoring post-reversal. This will help identify when volatility falls sharply after a reversal, potentially signaling that the downward movement is complete and new buying pressure may emerge, forcing remaining shorts to cover their positions in a buy cycle. This volatility component will help refine entry and exit timing for exploiting the resistance pattern phenomenon.
.