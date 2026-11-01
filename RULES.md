# Can't Stop — Rules Specification (FINAL)

Source of truth for the simulator. Engine and tests must match this file.
Any future rule change must be made here first, then in code and tests.

## Components
- 4 six-sided dice.
- 11 columns numbered 2..12 with lengths:
  `2:3, 3:5, 4:7, 5:9, 6:11, 7:13, 8:11, 9:9, 10:7, 11:5, 12:3`
- Each player has permanent progress per column (a position 0..length).
- 3 "runners" (neutral markers) shared by whoever's turn it is.
- 2–4 players (training default: 2).

## Turn structure
A turn is a sequence of rolls. Runner positions are provisional until the
player stops.

1. **Roll** the 4 dice.
2. **Pair** the dice into two pairs (3 distinct pairings of 4 dice). Each
   pair's sum is a column number. The player picks one pairing.
3. **Legal sums:** a sum is playable if its column is
   - not captured by any player (including self), AND
   - either already has a runner this turn, or a runner is still free
     (fewer than 3 runners placed), AND
   - the runner is not already at the top of the column.
4. **Placement rules:**
   - If both sums of the chosen pairing are playable, both are applied
     (a runner advances one step per sum; same sum twice = two steps,
     capped at the top). If both need a *new* runner but only one is free,
     only one can be placed — player's choice of which.
   - Playing both is mandatory: if both sums of the chosen pairing are
     playable, the player cannot apply just one. The only exception is the
     runner limit above, where only one can be placed.
   - If only one sum is playable, only that one is applied.
   - The player may choose ANY pairing that has at least one playable sum,
     even if another pairing would let them play two sums. (E.g. they may
     pick a pairing with one playable sum over one with two.)
   - If at least one pairing has a playable sum, the player must pick one of
     those; they can't decline to play and bust.
   - Pairings with no playable sum are not legal choices when any other
     pairing has one.
5. **Bust:** if NO pairing yields a playable sum, the turn ends and all runner
   progress this turn is lost. Permanent progress is unchanged. Turn passes.
6. **Decision:** after a successful roll the player chooses
   - **Continue** (roll again), or
   - **Stop**: runner positions become permanent progress; turn passes.
   The player may not stop before making their first successful roll
   (stop is only legal after at least one successful roll).

## Capturing columns
- A runner that reaches the top of a column is "at the top" for the rest of
  the turn (no further progress there).
- On **Stop**, every column whose runner is at the top is **captured** by
  that player. Captured columns are closed to everyone for the rest of the game.
- On **Bust**, nothing is captured.
- Captured columns do not count as one of the 3 runners in later turns
  (they simply can't be played).
- Two players can't both capture a column; the first to *stop* at the top wins
  it. (Turns are sequential, so no conflict arises.)

## Winning
First player to own **3 captured columns** at the end of their turn wins.

## Edge cases to be tested
- All 3 runners placed and no pairing hits them → bust.
- A sum whose column was captured earlier → not playable.
- Double sum on a column 1 step from the top → advance only 1, capture on stop.
- Runner at top: that column is not playable again that turn.
- Starting a runner from non-zero permanent progress.
- Bust discards runner progress and captures.
- Stop forbidden before first successful roll.
- Game ends immediately when a player's stop yields their 3rd capture.

