-- ─── PROFILES ────────────────────────────────────────────────────────────────
-- Extends auth.users with app-specific user data.
-- Row is auto-created via trigger on every new sign-up.

create table if not exists profiles (
    id          uuid primary key references auth.users (id) on delete cascade,
    full_name   text,
    device_metadata jsonb default '{}',
    created_at  timestamptz default now()
);

alter table profiles enable row level security;

create policy "users can view their own profile"
    on profiles for select
    using (auth.uid() = id);

create policy "users can update their own profile"
    on profiles for update
    using (auth.uid() = id);

-- Auto-create a profile row whenever a new auth user signs up
create or replace function handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
    insert into profiles (id, full_name)
    values (
        new.id,
        coalesce(new.raw_user_meta_data->>'full_name', '')
    );
    return new;
end;
$$;

create or replace trigger on_auth_user_created
    after insert on auth.users
    for each row execute procedure handle_new_user();


-- ─── PLANTS ──────────────────────────────────────────────────────────────────

create table if not exists plants (
    id          uuid primary key default gen_random_uuid(),
    created_by  uuid not null references profiles (id) on delete cascade,
    name        text not null,
    strain      text,
    status      text not null default 'active'
                    check (status in ('active', 'harvested', 'removed')),
    metadata    jsonb default '{}',
    created_at  timestamptz default now()
);

alter table plants enable row level security;

create policy "users can view their own plants"
    on plants for select
    using (auth.uid() = created_by);

create policy "users can insert their own plants"
    on plants for insert
    with check (auth.uid() = created_by);

create policy "users can update their own plants"
    on plants for update
    using (auth.uid() = created_by);

create policy "users can delete their own plants"
    on plants for delete
    using (auth.uid() = created_by);


-- ─── ANALYSES — add plant_id + performed_by ──────────────────────────────────
-- Both columns are nullable so existing rows are not affected.

alter table analyses
    add column if not exists plant_id     uuid references plants (id) on delete set null,
    add column if not exists performed_by uuid references profiles (id) on delete set null;

alter table analyses enable row level security;

create policy "users can view their own analyses"
    on analyses for select
    using (auth.uid() = performed_by);

create policy "users can insert their own analyses"
    on analyses for insert
    with check (auth.uid() = performed_by);
