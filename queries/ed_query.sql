------------------Query 1: Mortality Features---------------------
select * from factihealth.mimic.mortality_data;

------------------ Query 2: Mortality Data------------
select * from 
(
select 
distinct b.*,
        TO_DATE(REPLACE(REPLACE(b.edadmittime, '2130', '2024'),'2129', '2024'),'YYYY-MM-DD') AS edadmit_time
from 
    factihealth.mimic.ecg_model_dataset a
inner join 
    factihealth.mimic.mortality_data b on a.subject_id=b.subject_id
where date_part(year,edadmittime) in (2126,2127,2128,2129, 2130)
and date_part(year,ecgtime) in (2126,2127,2128, 2129, 2130)
and ecgtime between edadmittime and eddischargetime
order by ecgtime,a.subject_id
)
where edadmit_time between '{start_date}' AND '{end_date}';

----------------Query 3: ECG----------------
select * from 
(
select 
distinct a. subject_id,
a.gender,
a.anchor_age,
a.dod,
TO_DATE(REPLACE(REPLACE(b.edadmittime, '2130', '2024'),'2129', '2024'),'YYYY-MM-DD') AS edadmit_time,
TO_DATE(REPLACE(REPLACE(a.ecgtime, '2130', '2024'),'2129', '2024'),'YYYY-MM-DD') AS ecg_time,
a.bandwidth,
a.filtering,
a.rr_interval,
a.p_onset,
a.p_end,
a.qrs_onset,
a.qrs_end,
a.t_end,
a.p_axis,
a.qrs_axis,
a.t_axis,
a.target_variable
from 
factihealth.mimic.ecg_model_dataset a
inner join 
factihealth.mimic.mortality_data b on a.subject_id=b.subject_id
where date_part(year,b.edadmittime) in (2129, 2130)
and date_part(year,a.ecgtime) in (2129, 2130)
and ecgtime between b.edadmittime and b.eddischargetime
order by ecgtime,a.subject_id
);

----------------Query 4: ECG----------------
select * from 
(
select 
distinct a. subject_id,
a.gender,
a.anchor_age,
TO_DATE(REPLACE(REPLACE(b.edadmittime, '2130', '2024'),'2129', '2024'),'YYYY-MM-DD') AS edadmit_time,
TO_DATE(REPLACE(REPLACE(a.ecgtime, '2130', '2024'),'2129', '2024'),'YYYY-MM-DD') AS ecg_time,
a.bandwidth,
a.filtering,
a.rr_interval,
a.p_onset,
a.p_end,
a.qrs_onset,
a.qrs_end,
a.t_end,
a.p_axis,
a.qrs_axis,
a.t_axis
from 
factihealth.mimic.ecg_model_dataset a
inner join 
factihealth.mimic.mortality_data b on a.subject_id=b.subject_id
where date_part(year,b.edadmittime) in (2129, 2130)
and date_part(year,a.ecgtime) in (2129, 2130)
and ecgtime between b.edadmittime and b.eddischargetime
order by ecgtime,a.subject_id
)
where ecg_time between '{start_date}' AND '{end_date}';


----------------Query 5: Gout----------------

with 
ecg as (
select 
    * 
from 
    (
    select 
        distinct a. subject_id,
        a.gender,
        a.anchor_age,
        TO_DATE(REPLACE(REPLACE(b.edadmittime, '2130', '2024'),'2129', '2024'),'YYYY-MM-DD') AS edadmit_time,
        TO_DATE(REPLACE(REPLACE(a.ecgtime, '2130', '2024'),'2129', '2024'),'YYYY-MM-DD') AS ecg_time,
        a.bandwidth,
        a.filtering,
        a.rr_interval,
        a.p_onset,
        a.p_end,
        a.qrs_onset,
        a.qrs_end,
        a.t_end,
        a.p_axis,
        a.qrs_axis,
        a.t_axis
    from 
        factihealth.mimic.ecg_model_dataset a
    inner join 
        factihealth.mimic.mortality_data b on a.subject_id=b.subject_id
    where date_part(year,b.edadmittime) in (2129, 2130)
    and date_part(year,a.ecgtime) in (2129, 2130)
    and ecgtime between b.edadmittime and b.eddischargetime
    order by ecgtime,a.subject_id
    )
    where ecg_time between '{start_date}' AND '{end_date}'
),gout as 
(select 
    subject_id, 
    chiefcomplaint 
from 
    factihealth.mimic.ed_triage
where stay_id in (
                    select 
                        max(stay_id) 
                    from factihealth.mimic.ed_triage 
                    group by subject_id)
)
select 
    b.chiefcomplaint, 
    a.* 
from ecg a 
left join gout b 
on a.subject_id = b.subject_id;