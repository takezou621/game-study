# Use Case Specification for game-study (Fortnite AI English Coach)

## Overview

This document defines all user-facing use cases for the game-study application.
Each use case represents a specific scenario that the system must handle.

---

## UC-1xx: Main Pipeline Flows

### UC-101: Video Processing Pipeline
**Actor**: User
**Precondition**: Valid video file exists
**Flow**:
1. User starts application with `--input video --video path/to/video.mp4`
2. System loads ROI config, triggers config, system prompt
3. System initializes vision components (ROI extractor, OCR, YOLO)
4. System initializes trigger engine
5. System initializes dialogue client (optional)
6. System initializes voice client (optional with --voice)
7. System opens video capture
8. For each frame:
   - Extract ROIs from frame
   - Run vision detection (HP, Shield, Knocked, Storm)
   - Build game state
   - Evaluate triggers
   - Generate response if triggered
   - Output voice if enabled
9. System closes capture and cleanup
**Postcondition**: Session logs saved to output directory

### UC-102: Frame Processing
**Actor**: System (internal)
**Precondition**: Video capture is open
**Flow**:
1. Read frame from capture
2. Extract hp_shield ROI
3. Extract minimap ROI
4. Extract knocked_revive ROI
5. Extract weapon_ammo ROI
6. Process each ROI with appropriate detector
7. Update state builder with detection results
8. Get current game state
9. Determine movement state (combat/non_combat)
10. Evaluate triggers
11. Log state to session
**Postcondition**: State updated, trigger evaluated

---

## UC-2xx: Vision Detection Flows

### UC-201: HP Detection
**Actor**: System (internal)
**Precondition**: HP/Shield ROI extracted
**Flow**:
1. OCR detector processes hp_shield ROI
2. Extract numeric value from OCR result
3. Validate value is in range 0-100
4. Return {value, source: "ocr", confidence}
**Postcondition**: HP value available for state building

### UC-202: Shield Detection
**Actor**: System (internal)
**Precondition**: HP/Shield ROI extracted
**Flow**:
1. OCR detector processes hp_shield ROI
2. Extract shield value (second number in HUD)
3. Validate value is in range 0-100
4. Return {value, source: "ocr", confidence}
**Postcondition**: Shield value available for state building

### UC-203: Knocked Status Detection
**Actor**: System (internal)
**Precondition**: Knocked/Revive ROI extracted
**Flow**:
1. YOLO detector processes knocked_revive ROI
2. Check for "knocked" visual indicator
3. Return {value: True/False, source: "yolo", confidence}
**Postcondition**: Knocked status available for state building

### UC-204: Storm Status Detection
**Actor**: System (internal)
**Precondition**: Minimap ROI extracted
**Flow**:
1. Vision detector processes minimap ROI
2. Check for storm circle overlay
3. Determine if player is inside storm
4. Determine if storm is shrinking
5. Return {in_storm, is_shrinking, source, confidence}
**Postcondition**: Storm status available for state building

### UC-205: Weapon Detection
**Actor**: System (internal)
**Precondition**: Weapon/Ammo ROI extracted
**Flow**:
1. YOLO detector processes weapon ROI
2. Identify weapon type (AR, Shotgun, SMG, etc.)
3. OCR detector extracts ammo count
4. Return {name, ammo, source, confidence}
**Postcondition**: Weapon info available for state building

---

## UC-3xx: State Building Flows

### UC-301: Build Complete Game State
**Actor**: System (internal)
**Precondition**: Vision detections complete
**Flow**:
1. State builder receives all detection results
2. Update player.status.hp
3. Update player.status.shield
4. Update player.status.is_knocked
5. Update player.weapon.name
6. Update player.weapon.ammo
7. Update player.inventory.materials
8. Update world.storm.phase
9. Update world.storm.in_storm
10. Update world.storm.is_shrinking
11. Update session.inactivity_duration_ms
**Postcondition**: Complete game state available

### UC-302: Determine Movement State
**Actor**: System (internal)
**Precondition**: Game state built
**Flow**:
1. Check if HP < 50
2. Check if in_storm is True
3. If either is True: return "combat"
4. Otherwise: return "non_combat"
**Postcondition**: Movement state determined

### UC-303: State Reset
**Actor**: System (internal)
**Precondition**: Game session ended
**Flow**:
1. Call state_builder.reset()
2. All values reset to defaults
3. HP = 100, Shield = 0, is_knocked = False
4. Storm values = None/False
5. Inactivity = 0
**Postcondition**: State ready for new session

---

## UC-4xx: Trigger Evaluation Flows

### UC-401: Evaluate All Triggers
**Actor**: System (internal)
**Precondition**: Game state and movement state available
**Flow**:
1. Get current timestamp
2. Calculate inactivity duration
3. For each rule (sorted by priority):
   a. Check if enabled
   b. Evaluate all conditions
   c. Check cooldown
   d. Check combat suppression
   e. Get template for movement state
4. Return highest priority triggered rule
5. Update last triggered timestamp
**Postcondition**: Trigger result or None

### UC-402: P0 Low HP Trigger
**Actor**: System (internal)
**Precondition**: player.status.hp < 30
**Flow**:
1. Condition evaluated: hp < 30 → True
2. Rule priority = 0 (highest)
3. Get template for movement state:
   - combat: "Low HP! Find cover immediately!"
   - non_combat: "Your health is critical. Heal up before fighting."
4. Return trigger result
**Postcondition**: Low HP warning triggered

### UC-403: P0 Knocked Trigger
**Actor**: System (internal)
**Precondition**: player.status.is_knocked = True
**Flow**:
1. Condition evaluated: is_knocked == True → True
2. Rule priority = 0 (highest)
3. Get template for movement state:
   - combat: "You're knocked! Ping your location!"
   - non_combat: "Stay calm! Your teammates are coming."
4. Return trigger result
**Postcondition**: Knocked warning triggered

### UC-404: P0 Storm Damage Trigger
**Actor**: System (internal)
**Precondition**: world.storm.in_storm = True
**Flow**:
1. Condition evaluated: in_storm == True → True
2. Rule priority = 0 (highest)
3. Get template for movement state:
   - combat: "Get out of the storm! Now!"
   - non_combat: "You're taking storm damage. Move to safe zone."
4. Return trigger result
**Postcondition**: Storm warning triggered

### UC-405: P1 Storm Shrinking Trigger
**Actor**: System (internal)
**Precondition**: world.storm.is_shrinking = True, no P0 conditions met
**Flow**:
1. Check P0 conditions (hp < 30, knocked, in_storm) → all False
2. Condition evaluated: is_shrinking == True → True
3. Rule priority = 1
4. Get template for movement state:
   - combat: "Storm is moving! Get ready to rotate."
   - non_combat: "The storm is shrinking. Time to move."
5. Return trigger result
**Postcondition**: Rotation reminder triggered

### UC-406: P2 Weapon Learning Trigger
**Actor**: System (internal)
**Precondition**: player.weapon.new_weapon_detected = True, movement_state = "non_combat"
**Flow**:
1. Check P0, P1 conditions → all False
2. Condition evaluated: new_weapon_detected == True → True
3. Check combat suppression → suppressed in combat
4. Rule priority = 2
5. Get template:
   - combat: None (suppressed)
   - non_combat: "You picked up a new weapon! Great choice."
6. Return trigger result (only in non_combat)
**Postcondition**: Learning message triggered

### UC-407: P3 Small Talk Trigger
**Actor**: System (internal)
**Precondition**: session.inactivity_duration_ms > 30000, movement_state = "non_combat"
**Flow**:
1. Check P0, P1, P2 conditions → all False
2. Condition evaluated: inactivity > 30000 → True
3. Check combat suppression → suppressed in combat
4. Rule priority = 3 (lowest)
5. Get template:
   - combat: None (suppressed)
   - non_combat: "How's it going? Anything to practice?"
6. Return trigger result
**Postcondition**: Small talk triggered

### UC-408: Multiple P0 Triggers Simultaneously
**Actor**: System (internal)
**Precondition**: Multiple P0 conditions met (e.g., low HP AND in storm)
**Flow**:
1. Evaluate all P0 rules
2. All meet conditions
3. Return first P0 rule (sorted by rule order in config)
4. Only one trigger fires per frame
**Postcondition**: Single P0 trigger returned

---

## UC-5xx: Cooldown and Priority Flows

### UC-501: Cooldown Suppression
**Actor**: System (internal)
**Precondition**: Trigger recently fired
**Flow**:
1. First evaluation: trigger fires, last_triggered_ms updated
2. Immediate second evaluation: current_time - last_triggered < cooldown_ms
3. Rule is_on_cooldown returns True
4. Trigger suppressed
5. After cooldown period: trigger can fire again
**Postcondition**: Trigger suppressed during cooldown

### UC-502: Priority Preemption
**Actor**: System (internal)
**Precondition**: Lower priority trigger conditions met, higher priority fires
**Flow**:
1. P2 learning trigger conditions met
2. P0 low HP trigger conditions also met
3. Rules evaluated in priority order (P0 first)
4. P0 fires, returns immediately
5. P2 never evaluated (or ignored)
**Postcondition**: Higher priority trigger wins

### UC-503: Combat Suppresses Learning
**Actor**: System (internal)
**Precondition**: movement_state = "combat", P2/P3 conditions met
**Flow**:
1. Evaluate P2 weapon learning
2. movement_state == "combat" AND priority in [2, 3]
3. Rule suppressed by combat_suppress_priority setting
4. Evaluate P3 small talk
5. Same suppression applies
6. Only P0/P1 triggers can fire in combat
**Postcondition**: Learning triggers suppressed in combat

---

## UC-6xx: Response Generation Flows

### UC-601: Template-Only Response
**Actor**: System (internal)
**Precondition**: Trigger fired, no OpenAI client
**Flow**:
1. Get template from trigger result
2. Use template directly as response
3. Log response
4. Return response text
**Postcondition**: Template response delivered

### UC-602: OpenAI Enhanced Response
**Actor**: System (internal)
**Precondition**: Trigger fired, OpenAI client available
**Flow**:
1. Get template from trigger result
2. Build context from game state
3. Call OpenAI API with template as context
4. Receive enhanced response
5. Truncate to max_response_length if needed
6. Log response
7. Return response text
**Postcondition**: Enhanced response delivered

### UC-603: Combat Template Selection
**Actor**: System (internal)
**Precondition**: movement_state = "combat"
**Flow**:
1. Get template for "combat" key
2. Template should be short, urgent
3. Use phrases like "!", "immediately", "now"
4. Example: "Low HP! Find cover immediately!"
**Postcondition**: Urgent combat template selected

### UC-604: Non-Combat Template Selection
**Actor**: System (internal)
**Precondition**: movement_state = "non_combat"
**Flow**:
1. Get template for "non_combat" key
2. Template should be conversational, educational
3. Use complete sentences, suggestions
4. Example: "Your health is critical. Heal up before fighting."
**Postcondition**: Conversational template selected

### UC-605: Template is None
**Actor**: System (internal)
**Precondition**: Template for movement state is None
**Flow**:
1. Get template for movement state
2. Template is None (e.g., combat template for P2)
3. Rule skipped during evaluation
4. No trigger fires for this rule
**Postcondition**: Rule skipped

---

## UC-7xx: Voice Output Flows

### UC-701: Voice Output with TTS API
**Actor**: System (internal)
**Precondition**: --voice flag, use_realtime_api = False
**Flow**:
1. Generate response text
2. Call OpenAI TTS API with text
3. Receive audio data
4. Queue audio for playback
5. Track duration for cooldown
**Postcondition**: Audio played to user

### UC-702: Voice Output with Realtime API
**Actor**: System (internal)
**Precondition**: --voice flag, use_realtime_api = True
**Flow**:
1. Establish WebSocket connection to OpenAI Realtime API
2. Send text or audio input
3. Receive streaming audio response
4. Play audio chunks as received
5. Lower latency than TTS API
**Postcondition**: Real-time audio played

### UC-703: Voice Interruption (P0 Preemption)
**Actor**: System (internal)
**Precondition**: Voice playing, P0 trigger fires
**Flow**:
1. P2 voice response is playing
2. P0 trigger fires (e.g., player knocked)
3. interrupt_requested = True
4. Current speech interrupted
5. P0 voice response starts immediately
6. P0 has highest priority
**Postcondition**: P0 voice preempts current speech

### UC-704: Combat Short Templates
**Actor**: System (internal)
**Precondition**: movement_state = "combat", voice output enabled
**Flow**:
1. P0 trigger fires in combat
2. Use COMBAT_TEMPLATES for short responses
3. Example: "Low HP! Cover!" instead of full sentence
4. Faster to speak, more urgent
**Postcondition**: Short, urgent voice response

### UC-705: Voice Cooldown
**Actor**: System (internal)
**Precondition**: Voice recently spoken
**Flow**:
1. Voice response just completed
2. last_spoken_time updated
3. Current time - last_spoken_time < cooldown_ms
4. New voice request queued or ignored
5. After cooldown: voice available again
**Postcondition**: Voice cooldown enforced

---

## UC-8xx: Error Handling Flows

### UC-801: Video File Not Found
**Actor**: User
**Precondition**: Invalid video path
**Flow**:
1. User specifies --video /non/existent.mp4
2. VideoFileCapture.__init__ checks file exists
3. FileNotFoundError raised
4. Application exits with error code 1
5. Error logged to session
**Postcondition**: Application exits gracefully

### UC-802: Missing Config File
**Actor**: User
**Precondition**: Invalid config path
**Flow**:
1. User specifies --triggers /non/existent.yaml
2. TriggerEngine.__init__ tries to load file
3. FileNotFoundError raised
4. Application exits with error code 1
5. Error logged
**Postcondition**: Application exits gracefully

### UC-803: Missing OpenAI API Key
**Actor**: User
**Precondition**: OPENAI_API_KEY not set, --voice or OpenAI requested
**Flow**:
1. Application starts
2. OpenAIClient initialization attempted
3. ValueError: "OpenAI API key not found"
4. Warning logged
5. Continue in template-only mode
6. Voice client disabled
**Postcondition**: Application continues without AI enhancement

### UC-804: Vision Detection Failure
**Actor**: System (internal)
**Precondition**: OCR/YOLO processing fails
**Flow**:
1. OCR detector processes ROI
2. Exception raised (e.g., invalid image format)
3. Exception caught in frame processing loop
4. Error logged with log_error()
5. Continue to next frame
6. Use default state values
**Postcondition**: Processing continues with defaults

### UC-805: OpenAI API Error
**Actor**: System (internal)
**Precondition**: OpenAI API call fails
**Flow**:
1. OpenAI client calls generate_response()
2. API returns error (rate limit, network, etc.)
3. Exception caught
4. Warning logged
5. Fall back to template response
6. Continue processing
**Postcondition**: Template fallback used

### UC-806: WebSocket Connection Lost
**Actor**: System (internal)
**Precondition**: Realtime API in use, connection drops
**Flow**:
1. WebSocket connection to OpenAI lost
2. Connection error detected
3. Attempt reconnection (with retry)
4. If reconnection fails: fall back to TTS API
5. If TTS fails: use text-only mode
**Postcondition**: Graceful degradation

---

## UC-9xx: Session Management Flows

### UC-901: Session Logging
**Actor**: System (internal)
**Precondition**: Session started
**Flow**:
1. SessionLogger initialized with output directory
2. State logged to state.jsonl each frame
3. Triggers logged to triggers.jsonl
4. Responses logged to responses.jsonl
5. Errors logged to errors.log
**Postcondition**: Complete session audit trail

### UC-902: Sensitive Data Masking
**Actor**: System (internal)
**Precondition**: Log contains sensitive data
**Flow**:
1. SensitiveFormatter applied to logs
2. API keys pattern matched
3. Replaced with ***REDACTED***
4. Bearer tokens masked
5. Base64-like strings masked
**Postcondition**: Logs safe for sharing

### UC-903: Health Check
**Actor**: Admin
**Precondition**: Application running
**Flow**:
1. check_health() called
2. Check config files exist
3. Check directories writable
4. Check API key configured (optional)
5. Return {healthy: bool, details: {...}}
**Postcondition**: Health status available

### UC-904: Metrics Collection
**Actor**: System (internal)
**Precondition**: Application running
**Flow**:
1. MetricsCollector tracks counters
2. frames_processed incremented each frame
3. triggers_fired incremented each trigger
4. api_calls incremented each API call
5. latency_samples recorded
6. get_summary() returns aggregated metrics
**Postcondition**: Metrics available for monitoring

---

## UC-10xx: Edge Case Flows

### UC-1001: Zero HP
**Actor**: System (internal)
**Precondition**: HP detected as 0
**Flow**:
1. OCR detects HP = 0
2. Condition hp < 30 evaluated
3. 0 < 30 is True
4. P0 low HP trigger fires
**Postcondition**: Warning triggered for zero HP

### UC-1002: Confidence Out of Range
**Actor**: System (internal)
**Precondition**: Vision returns invalid confidence
**Flow**:
1. Vision detection returns confidence = 1.5
2. StateValueModel validates confidence
3. ValueError raised: "must be between 0.0 and 1.0"
4. Caller must handle exception
**Postcondition**: Invalid value rejected

### UC-1003: Empty/Corrupted Frame
**Actor**: System (internal)
**Precondition**: Frame read returns None or corrupted
**Flow**:
1. capture.read() returns None
2. Frame loop breaks
3. Processing ends gracefully
4. Session logs saved
**Postcondition**: Graceful end of processing

### UC-1004: Rapid State Changes
**Actor**: System (internal)
**Precondition**: State changes rapidly (100+ updates/second)
**Flow**:
1. State builder receives rapid updates
2. Each update overwrites previous value
3. Latest value used for trigger evaluation
4. No race conditions (single-threaded)
**Postcondition**: Final state is last update

### UC-1005: All Triggers On Cooldown
**Actor**: System (internal)
**Precondition**: Multiple triggers recently fired
**Flow**:
1. All applicable triggers are on cooldown
2. evaluate_triggers() returns None
3. No response generated
4. Processing continues
**Postcondition**: Silent frame (no trigger)

---

## Summary Table

| Category | Use Cases | Description |
|----------|-----------|-------------|
| Main Pipeline | UC-101 to UC-102 | Video processing, frame loop |
| Vision | UC-201 to UC-205 | HP, Shield, Knocked, Storm, Weapon detection |
| State | UC-301 to UC-303 | State building, movement state, reset |
| Triggers | UC-401 to UC-408 | All trigger evaluation scenarios |
| Cooldown/Priority | UC-501 to UC-503 | Cooldown, preemption, combat suppression |
| Response | UC-601 to UC-605 | Template, OpenAI enhancement, selection |
| Voice | UC-701 to UC-705 | TTS, Realtime API, interruption |
| Error Handling | UC-801 to UC-806 | Missing files, API errors, fallbacks |
| Session | UC-901 to UC-904 | Logging, masking, health, metrics |
| Edge Cases | UC-1001 to UC-1005 | Zero HP, confidence, empty frames |
