# Reading a chess network's mind in plain chess rules: what Rosetta found

## The problem

Chess networks like Leela Chess Zero play far better than people, but what they know is locked inside millions of
numbers. Researchers can check whether a network contains concepts they already have names for, such as material or
king safety. What they cannot easily do is read off the network's own ideas, test whether those ideas are really what
the network uses, and say what is left that nobody has a name for.

## What we built

Rosetta is a translator. For each variable inside the network it searches for a short chess rule (at most three
conditions, such as "an enemy piece, attacked more times than it is defended, on the square my move goes to") that
describes when the variable is active. Then it checks the rule the hard way:

- it swaps the rule in for the variable inside the network and checks the network still behaves the same;
- it checks the rule against real game results, including 16,000 games it never saw;
- it checks that a second, independently trained network has the same variable.

The rules come from a fixed chess vocabulary plus the network's own planned line of play, so Rosetta can also notice
when a network is thinking ahead. A rule cannot hide a copy of the network inside it: no learned numbers are allowed.

## What we found

1. **It finds hidden rules without being told.** On three test networks trained on secret rules, it recovered all
   three exactly or in an equivalent form.

2. **Chess networks think ahead mostly along forcing lines, and the transformer thinks ahead about three times as
   hard.** Erasing the network's memory of the square where its next-but-one move will land makes it less sure of its
   current move. In all three Leela networks this happens mainly when that future move is a check or a capture, and
   most strongly in the transformer. For quiet future moves the effect is small (the transformer and one convolutional
   network) or absent (the other). This was checked again, with predictions written down first, on 2,000 positions
   from games never used before; one of the three predictions only partly held. (The transformer is also a stronger
   network, so this may be about strength as well as design.)

3. **One of the network's own ideas passes every test we set in advance:** a detector for "the move I am about to
   play wins material: the piece on its target is outnumbered or undefended". Swapped into the network, the rule does
   about half of the variable's job. All three Leela networks we opened have this variable, and in each one the rule
   can be swapped in for it (it does 50% to 55% of the job in the two convolutional networks and 73% in the
   transformer). It is a known chess idea, now shown to be a real, working part of
   these networks.

4. **The translated rules carry real knowledge about who wins.** Added to everything textbook chess measures, they
   improve predictions of game results on 16,000 unseen games, by about a fifth of what the network itself adds. That
   knowledge is spread over many rules; no handful of them carries it.

5. **A full dictionary of one layer.** 485 of the 512 variables in Leela's last layer got a chess rule. Rebuilding
   that layer from the rules alone, the network still picks the same move 41% of the time (5% for a layer that knows
   nothing). But it loses almost all of its sense of who is winning.

6. **The network's judgement runs through about 16 of its 512 internal variables, and a third of what it knows beyond
   the textbook is what its own planned moves win.** Of what the network knows about who wins beyond textbook chess,
   91% passes through about 16 of the 512 variables in its last layer (16 picked at random carry 13% to 62%); in a
   second Leela network the same holds (78%). Those 16 split into "we are winning", "we are losing" and "this is
   drawish" signals rather than one score. Our
   rules describe only 2 of those 16, which is why the rebuilt layer loses the evaluation. One part we can name: the
   material the network wins along the three moves it plans ahead accounts for a third of its extra knowledge (the
   same count along random moves accounts for almost none).

## What we don't have yet

- **No new chess strategy.** The part of the network's judgement that textbook chess misses is still mostly
  unexplained; the translated rules close about 7% of it.
- **The rules capture the network's move logic far better than its judgement.** We now know where the judgement
  lives (about 16 variables) but not yet what they compute.
- **The passing concept passes at the margin**, and only after combining three rules.
- **The look-ahead comparison** involves three networks, which differ in strength as well as design.

## Next

Translate the transformer (the strongest network, and the one that looks ahead) layer by layer, to show its
look-ahead as an explicit step. Then test the translated rules on people: do players who learn them choose better
moves?

Code, data and every number: branch `rosetta` of github.com/cybercryptixcoder/glass_nets, `docs/rosetta/` and
`results/rosetta/`.
