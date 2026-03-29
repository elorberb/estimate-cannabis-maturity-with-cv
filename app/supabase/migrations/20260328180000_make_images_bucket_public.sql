insert into storage.buckets (id, name, public)
values ('images', 'images', true)
on conflict (id) do update set public = true;

create policy "images_public_read"
on storage.objects for select
using (bucket_id = 'images');
