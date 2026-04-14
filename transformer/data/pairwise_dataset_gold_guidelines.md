# Pairwise Gold Annotation Guidelines

## Goal

Annotate pairwise accident-statement examples for:

-> `gold_label`
-> `gold_inconsistency_type`
-> `rationale_short`

## Relation Labels

Use exactly one:

-> `supports`: the two texts describe compatible versions of the same event
-> `neutral`: the two texts do not clearly support each other and do not clearly contradict each other
-> `contradicts`: the two texts contain a meaningful factual conflict

## Inconsistency Types

Use exactly one:

-> `none`
-> `damage_mismatch`
-> `dynamics_mismatch`
-> `phantom_vehicle`

Rules:

-> if `gold_label != contradicts`, then `gold_inconsistency_type` must be `none`
-> if `gold_label == contradicts`, choose the single strongest inconsistency type

Definitions:

-> `damage_mismatch`: the statements conflict about what damage exists or what visually happened to the vehicle
-> `dynamics_mismatch`: the statements conflict about motion, impact direction, traffic light state, who hit whom, stopping vs moving, lane behavior, or general accident mechanics
-> `phantom_vehicle`: one story depends on a missing or unsupported extra vehicle that the other story or context does not ground

## Rationale

Write one short sentence describing the main reason for the chosen label.

Good examples:

-> `One text says the insured was stopped and hit from behind, while the other says the insured rolled backward into the other vehicle.`
-> `Both texts describe the same rear-end collision and the same visible rear/front damage.`
-> `The witness only reports the aftermath and does not confirm the insured's version of the collision dynamics.`

## Annotation Policy

-> annotate from the texts and metadata shown in the template
-> do not use the hidden weak labels from the metadata file as annotation truth
-> prefer `neutral` over `contradicts` when the conflict is not explicit enough
-> prefer the most concrete contradiction type when `gold_label = contradicts`
