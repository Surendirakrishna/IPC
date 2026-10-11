"""Apply WIPO priority rules between candidates that each passed the evidence policy.

Rules come from the version-matched scheme notes, title references and definitions attached to each
candidate (its own and its ancestors'). They are applied according to what they say:

- precedence: when X and Y are both supported and Y is a preferred place for a rule whose dominated scope
  covers X, X becomes an alternative. A rule never adds Y: Y must itself be a supported candidate.
  Conditional rules ("For ...", "If ...", "e.g.") only add a review note, because the condition is
  not checked automatically.
- place_last / place_first: between two supported candidates in the rule's scope (neither an ancestor
  of the other), the later / earlier place in scheme order is kept.
- limiting_reference: if another supported candidate is a cited place, both are kept with a review note.

Indexing notes, definitions, scope statements and informative references are context only.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from app.services.ingestion.wipo import display_symbol


@dataclass
class RuleCandidate:
    code: str
    display_code: str
    ancestor_codes: list[str]
    official_context: list[dict[str, Any]]

    @property
    def chain(self) -> set[str]:
        return {self.code, *self.ancestor_codes}

    def under(self, symbols: set[str]) -> bool:
        return not self.chain.isdisjoint(symbols)


@dataclass
class RuleOutcome:
    demoted: dict[str, str] = field(default_factory=dict)  # code -> reason
    notes: dict[str, list[str]] = field(default_factory=dict)  # code -> review notes


def _source(item: dict[str, Any]) -> str:
    owner = display_symbol(item["owner_symbol"])
    if item.get("scope_end_symbol"):
        owner += f"-{display_symbol(item['scope_end_symbol'])}"
    kind = {"scheme_note": "scheme note", "scheme_reference": "scheme title reference"}.get(
        item["source_kind"], f"definition ({item['part_type']})")
    edition = f", edition {item['editions']}" if item.get("editions") else ""
    return f"WIPO {kind} for {owner}{edition}"


def _names(cands: list[RuleCandidate]) -> str:
    return ", ".join(c.display_code for c in cands)


def apply_scheme_rules(
    candidates: list[RuleCandidate],
    others: list[RuleCandidate],
    *,
    precedence: bool,
    place: bool,
) -> RuleOutcome:
    """candidates: supported candidates in rank order; others: ranked but not supported."""
    beats: dict[str, dict[str, str]] = defaultdict(dict)  # loser -> {winner: reason}
    notes: dict[str, list[str]] = defaultdict(list)
    for x in candidates:
        for item in x.official_context:
            for rule in item.get("rules") or []:
                kind = rule.get("kind")
                text = rule.get("sentence") or item.get("text") or ""
                source = _source(item)
                if kind == "precedence":
                    preferred, dominated = set(rule["preferred"]), set(rule["dominated"])
                    if not x.under(dominated) or x.under(preferred):
                        continue
                    winners = [y for y in candidates if y.code != x.code and y.under(preferred)]
                    if winners and precedence and not rule.get("conditional"):
                        for y in winners:
                            beats[x.code].setdefault(
                                y.code, f"{source}: \"{text}\" ({y.display_code} takes precedence over {x.display_code})."
                            )
                    elif winners:
                        why = "conditional rule, confirm the condition" if rule.get("conditional") else "not applied"
                        notes[x.code].append(f"WIPO precedence ({why}): {source}: \"{text}\" "
                                             f"({_names(winners)} would take precedence over {x.display_code}).")
                    else:
                        ranked = [y for y in others if y.under(preferred)]
                        if ranked:
                            notes[x.code].append(
                                f"WIPO precedence: {source}: \"{text}\". {_names(ranked)} was ranked but not "
                                f"supported by the evidence policy; if its subject matter is also disclosed, it takes "
                                f"precedence over {x.display_code}."
                            )
                elif kind in ("place_last", "place_first"):
                    scope = set(rule["scope"])
                    if not x.under(scope):
                        continue
                    last = kind == "place_last"
                    for y in candidates:
                        if (y.code == x.code or not y.under(scope) or y.code in x.ancestor_codes
                                or x.code in y.ancestor_codes):
                            continue
                        if (y.code > x.code) != last:
                            continue
                        reason = (f"{source}: \"{text}\" ({y.display_code} is the {'last' if last else 'first'} "
                                  f"appropriate place, not {x.display_code}).")
                        if place:
                            beats[x.code].setdefault(y.code, reason)
                        else:
                            notes[x.code].append(f"WIPO place rule (not applied): {reason}")
                elif kind == "limiting_reference":
                    targets = set(rule["targets"])
                    if x.under(targets):
                        continue
                    cited = [y for y in candidates if y.code != x.code and y.under(targets)]
                    if cited:
                        notes[x.code].append(
                            f"WIPO limiting reference: {source}: \"{text}\". The subject matter named there is "
                            f"classified in {_names(cited)}; check which place this disclosure belongs to."
                        )

    out = RuleOutcome()
    for x in candidates:
        # A loser stays if every rule it loses by is mutual (conflicting rules are left for review).
        reasons = [reason for winner, reason in beats.get(x.code, {}).items() if x.code not in beats.get(winner, {})]
        if reasons:
            out.demoted[x.code] = reasons[0]
        elif beats.get(x.code):
            notes[x.code].append("Conflicting WIPO priority rules between candidates; kept for review: "
                                 + " | ".join(beats[x.code].values()))
    out.notes = {code: list(dict.fromkeys(lst)) for code, lst in notes.items()}
    return out
