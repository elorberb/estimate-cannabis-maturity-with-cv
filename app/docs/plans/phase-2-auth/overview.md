# Phase 2: User Accounts

## Goal

Add authentication so users can sign in, own their analysis history, and access it from any device. Seamless migration from anonymous to authenticated usage.

## Scope

- Supabase Auth with Email + Google + Apple Sign-In
- Auth screen (login/signup) + Profile screen
- JWT middleware for FastAPI (backward compatible — anonymous still works)
- Device-to-user migration flow for anonymous history
- Row-level security policies on analyses table

## Auth Methods

- **Email + password** — standard signup/login
- **Google Sign-In** — OAuth via Supabase
- **Apple Sign-In** — required by App Store if any social login is offered

## New Screens

**Screen 7: Auth Screen (Login/Signup)**
- Tab toggle: Login | Sign Up
- Email + password fields
- Social login buttons: "Continue with Google", "Continue with Apple"
- "Skip for now" link (anonymous mode continues to work)

**Screen 8: Profile Screen**
- User info (email, name)
- Total analyses count
- Account settings (change password, delete account)
- Logout button

## Migration Flow

When a user who has been using the app anonymously signs up:
1. App sends `POST /api/v1/auth/migrate` with `device_id`
2. Backend updates all `analyses` where `device_id` matches and `user_id` is NULL
3. Sets `user_id` to the new authenticated user
4. User sees all their previous analyses in their account

## Data Model Changes

See [Data Model](../2026-03-07-data-model.md#phase-2-auth-additions) for schema changes.

## API Changes

See [Data Model](../2026-03-07-data-model.md#phase-2-auth) for endpoint details.

## Duration

Weeks 5-7

## Dependencies

Requires Phase 1 complete.
