---
title: Chat settings
description: Configure how the agent behaves on the web chat channel.
---

# Chat settings

<p class="lead">
Chat settings configure how the agent behaves on the web chat channel.
They are defined in <code>chat/configuration.yaml</code>.
</p>

!!! note "Webchat configs must all be present together"
    `ChatGreeting`, `ChatStylePrompt`, and `ChatSafetyFilters` (the three webchat config resources) follow an all-or-nothing rule: if any one of them is present in your project, all three must be present. Pushing only a subset raises a validation error listing the missing types.

!!! info "Pushing new webchat configs enables the webchat channel automatically"
    When you push webchat config resources for the first time, the ADK automatically enables the webchat channel on the platform. You do not need to enable it manually in Agent Studio. The configs are sent as updates internally, so the push behaves correctly whether or not the channel was previously enabled.

## Location

~~~text
chat/
├── configuration.yaml
└── safety_filters.yaml       # Optional
~~~

## What chat settings control

<div class="grid cards" markdown>

-   **Greeting**

    ---

    The first message the agent sends when a chat session starts.

-   **Style prompt**

    ---

    Channel-specific instructions that shape how the agent writes in chat.

-   **Safety filters**

    ---

    Optional chat-channel content safety filter overrides.

</div>

## Greeting

The greeting is the first message the agent sends when a chat session starts.

### Fields

| Field | Required | Description |
|---|---|---|
| `welcome_message` | Yes | Text of the greeting. Supports `{{attr:...}}` and `{{vrbl:...}}` references. |
| `language_code` | Yes | BCP-47 language code, for example `en-GB` or `en-US`. |

### Example

~~~yaml
greeting:
  welcome_message: Hi there! How can I help you today?
  language_code: en-GB
~~~

## Style prompt

The style prompt contains channel-specific instructions that shape how the agent writes.

Use this for chat-specific guidance such as:

- keeping responses concise
- using bullet points for lists
- adjusting formatting for readability

This is separate from the agent's broader personality. Use it to control how the agent communicates specifically in web chat.

### Fields

| Field | Required | Description |
|---|---|---|
| `prompt` | No | Free-text style instructions. Resource references are not allowed. |

### Example

~~~yaml
style_prompt:
  prompt: You are a helpful and professional web chat assistant. Keep responses concise and use formatting where appropriate.
~~~

!!! tip "Keep chat guidance channel-specific"

    Use the style prompt for instructions that only apply to chat, such as formatting, brevity, or written tone. Use agent settings for broader identity and behavioral guidance.

## Safety filters

`chat/safety_filters.yaml` is an optional file that overrides the project-level safety filter settings for the chat channel. When present, it takes precedence over `agent_settings/safety_filters.yaml` for chat interactions.

See the [Safety filters reference](./safety_filters.md) for the full schema, field descriptions, and examples.

## Full example

~~~yaml
greeting:
  welcome_message: Hi! How can I help you today?
  language_code: en-GB
style_prompt:
  prompt: You are a helpful and professional web chat assistant. Keep responses concise.
~~~

## Related pages

<div class="grid cards" markdown>

-   **Safety filters**

    ---

    Configure content safety filtering at the project and channel level.
    [Open safety filters](./safety_filters.md)

-   **Agent settings**

    ---

    Define the agent's overall identity, role, and rules.
    [Open agent settings](./agent_settings.md)

-   **Voice settings**

    ---

    Configure the equivalent behavior for the voice channel.
    [Open voice settings](./voice_settings.md)

</div>
