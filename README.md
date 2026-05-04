# CyberNoodles
A local-run AI that can play Beat Saber!?

# Chores

- [ ] Make the documentation as to how I made this, and what everything does (I don't want to do this part)
- [ ] Make some replays to show off it's skill
- [ ] Use PyOxidizer to make it into one simple RUST .exe (Migrating some code to Rust - WIP)
- [ ] Make premade model for base (Training Model - WIP)
- [ ] Crouching and dodging walls with head (Implemented head movement - WIP)
- [ ] Realistic saber movement (Implemented - WIP)
- [ ] Make premade model for base (Training Model - WIP)
- [ ] Make a standalone RUST exe for just inference (Low Priority)

# Completed chores/fixes

- [x] Fix shard schema (I am on Schema v15 - Prototype, prep for Schema v1 - Release)
- [x] Fix BC not learning properly
- [x] Fix AWAC just regularizing into a BC model?!
- [x] Fix hits not resolving properly
- [x] Fix default flags (Default to Rust, not to Python)
- [x] Implement Shard processing into Rust! (Wow, a MASSIVE speed boost with 8 workers...) (From 60 replays/min to 1200 replays/min!!!)
- [x] Implement BC patience to prevent overfitting
- [x] Migrate from BC -> RL into BC -> AWAC -> RL, for a much nicer model outcome, I hope...
- [x] Put files into folders to make it more neat
- [x] Add anti-abuse to replays (Encode some crap at the end of the replay so people can't pass it off as their own play)
- [x] Make process more streamlined, add instruction within the program, so they don't have to refer back to here
- [x] Migrate code that can be in Rust, to Rust
- [x] Make standalone script for pure inference (No training, for the basic people)
