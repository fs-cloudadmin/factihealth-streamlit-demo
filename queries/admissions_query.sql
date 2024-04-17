---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------

-- Description: Below code is used to fetch data from mimic admissions dataset. The masked data has been differed by 100 years
--              to get the data from year 2115 to 2015.         

---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------

--select 
--   subject_id, 
--   hadm_id, 
--   dateadd(year,-100,admittime) as admittime, 
--   dateadd(year,-100,dischtime) as dischtime, 
--   case 
--       when (lower(admission_location) like '%transfer%') then dateadd(year,-91,admittime)
--       when (lower(admission_location) not like '%transfer%') then null 
--   end as transtime, 
--   admission_type,
--   admission_location, 
--   discharge_location,
--   marital_status, 
--   race  
--from
--   (
--   select distinct *  
--   from factihealth.mimic.admissions
--   where (edregtime is null and edouttime is null)
--   and date_part(year,admittime) = 2115
--   and date_part(month,admittime) = 02
--   order by subject_id
--   limit 50
--   );

---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------

-- Description: Below code is used to fetch data from mimic admissions dataset. The masked data has been differed by 100 years
--              to get the data from year 2115 to 2015.         

---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------

select distinct * from
(
select
subject_id, 
hadm_id, 
TO_DATE(REPLACE(REPLACE(admittime, '2130', '2024'),'2129','2024'),'YYYY-MM-DD') AS admittime,
TO_DATE(REPLACE(dischtime, '2130', '2024'),'YYYY-MM-DD') AS dischtime,
--dateadd(year,-107,admittime) as admittime, 
--dateadd(year,-107,dischtime) as dischtime, 
case 
    when (lower(admission_location) like '%transfer%') then TO_DATE(REPLACE(REPLACE(admittime, '2130', '2024'),'2129','2024'),'YYYY-MM-DD')
    when (lower(admission_location) not like '%transfer%') then null 
end as transtime, 
admission_type,
admission_location, 
discharge_location,
marital_status, 
race  
from
   (
   select distinct *  
   from factihealth.mimic.admissions
   where (edregtime is null and edouttime is null)
   --and (admittime >= '2130-03-01' and admittime <= '2130-03-04')
   )
)
where (admittime >= '{old_start_date}' and  admittime <= '2024-03-04')
order by admittime
;