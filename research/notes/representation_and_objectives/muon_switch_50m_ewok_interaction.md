# fw ewok interaction reader — FW EWoK interaction reader

Post-endpoint four-cell readout for the FW shared-anchor models. It scores EWoK contexts and targets directly and measures stable conditional-reversal failures without using old baseline correctness columns.

- `adamw50`: accuracy=0.4923864531373064, wrong=3867, stable failures=2702 (0.6987328678562192 of wrong), interaction median wrong=-0.9083619751036167
- `continuous_muon50`: accuracy=0.5174586505644526, wrong=3676, stable failures=2484 (0.675734494015234 of wrong), interaction median wrong=-0.761275626718998
- `muon20toadamw50`: accuracy=0.4997374639012864, wrong=3811, stable failures=2648 (0.6948307530831803 of wrong), interaction median wrong=-0.9959949052426964
- `muon40toadamw50`: accuracy=0.4977684431609346, wrong=3826, stable failures=2639 (0.6897543125980136 of wrong), interaction median wrong=-0.8240304756909609

Combined summary: `experiments/archive/representation_and_objectives/data/muon_switch_50m_ewok_interaction/fw_ewok_interaction_reader_summary.json`
