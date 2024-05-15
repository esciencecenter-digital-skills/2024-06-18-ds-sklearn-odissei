---
layout: workshop      # DON'T CHANGE THIS.
# More detailed instructions (including how to fill these variables for an
# online workshop) are available at
# https://carpentries.github.io/workshop-template/customization/index.html
venue: "FIXME"        # brief name of the institution that hosts the workshop without address (e.g., "Euphoric State University")
address: "FIXME"      # full street address of workshop (e.g., "Room A, 123 Forth Street, Blimingen, Euphoria"), videoconferencing URL, or 'online'
country: "FIXME"      # lowercase two-letter ISO country code such as "fr" (see https://en.wikipedia.org/wiki/ISO_3166-1#Current_codes) for the institution that hosts the workshop
language: "FIXME"     # lowercase two-letter ISO language code such as "fr" (see https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) for the
latitude: "45"        # decimal latitude of workshop venue (use https://www.latlong.net/)
longitude: "-1"       # decimal longitude of the workshop venue (use https://www.latlong.net)
humandate: "FIXME"    # human-readable dates for the workshop (e.g., "Feb 17-18, 2020")
humantime: "FIXME"    # human-readable times for the workshop (e.g., "9:00 am - 4:30 pm")
startdate: FIXME      # machine-readable start date for the workshop in YYYY-MM-DD format like 2015-01-01
enddate: FIXME        # machine-readable end date for the workshop in YYYY-MM-DD format like 2015-01-02
instructor: ["instructor one", "instructor two"] # boxed, comma-separated list of instructors' names as strings, like ["Kay McNulty", "Betty Jennings", "Betty Snyder"]
helper: ["helper one", "helper two"]     # boxed, comma-separated list of helpers' names, like ["Marlyn Wescoff", "Fran Bilas", "Ruth Lichterman"]
email: ["training@esciencecenter.nl"]    # boxed, comma-separated list of contact email addresses for the host, lead instructor, or whoever else is handling questions, like ["marlyn.wescoff@example.org", "fran.bilas@example.org", "ruth.lichterman@example.org"]
collaborative_notes:  # optional: URL for the workshop collaborative notes, e.g. an Etherpad or Google Docs document (e.g., https://pad.carpentries.org/2015-01-01-euphoria)
eventbrite:           # optional: alphanumeric key for Eventbrite registration, e.g., "1234567890AB" (if Eventbrite is being used)
---

{% comment %} See instructions in the comments below for how to edit specific sections of this workshop template. {% endcomment %}

{% comment %}
HEADER

Edit the values in the block above to be appropriate for your workshop.
If the value is not 'true', 'false', 'null', or a number, please use
double quotation marks around the value, unless specified otherwise.
And run 'make workshop-check' *before* committing to make sure that changes are good.
{% endcomment %}


{% comment %}
8< ============= For a workshop delete from here =============
For a workshop please delete the following block until the next dashed-line
{% endcomment %}

{% comment %}
  Assign first row in data file as info to access workshop data
{% endcomment %}
{% assign info = site.data.data[0] %}

{% comment %}
  Assign value in eventbrite file as eventbrite to access the code
{% endcomment %}
{% assign eventbrite = site.data.eventbrite %}

{% comment %}
<div class="alert alert-danger">
This is the workshop template. Delete these lines and use it to
<a href="https://carpentries.github.io/workshop-template/customization/index.html">customize</a>
your own website. If you are running a self-organized workshop or have not put
in a workshop request yet, please also fill in
<a href="{{site.amy_site}}/forms/self-organised/">this workshop request form</a>
to let us know about your workshop and our administrator may contact you if we
need any extra information.
If this is a pilot workshop for a new lesson,
remember to uncomment the `pilot_lesson_site` field in `_config.yml`
</div>
{% endcomment %}

{% comment %}
8< ============================= until here ==================
{% endcomment %}

{% comment %}
Check carpentry
{% endcomment %}

{% if info.carpentry == "FIXME" %}
<div class="alert alert-warning">
It looks like you are setting up a website for a workshop but you haven't specified the carpentry type in the <code>_data/data.csv</code> file (current value in <code>_data/data.csv</code>: "<strong>{{ info.carpentry }}</strong>", possible values: <code>swc</code>, <code>dc</code>, <code>lc</code>, <code>cp</code>, <code>ds</code>, or <code>pilot</code>). After editing this file, you need to run <code>make serve</code> again to see the changes reflected.
</div>
{% endif %}

{% comment %}
Read correct lesson meta from esciencecenter-digital-skills/workshop-metadata
{% endcomment %}


{% if info.flavor and info.flavor != 'NA' %}
{% capture lesson_meta %}https://raw.githubusercontent.com/esciencecenter-digital-skills/workshop-metadata/main/{{info.curriculum}}-{{info.flavor}}{% endcapture %}
{% else %}
{% capture lesson_meta %}https://raw.githubusercontent.com/esciencecenter-digital-skills/workshop-metadata/main/{{info.curriculum}}{% endcapture %}
{% endif %}


<h2 id="general">General Information</h2>

{% comment %}
INTRODUCTION

Edit the general explanatory paragraph below if you want to change
the pitch.
{% endcomment %}
{% if info.carpentry == "swc" %}
{% include swc/intro.html %}
{% elsif info.carpentry == "dc" %}
{% include dc/intro.html %}
{% elsif info.carpentry == "lc" %}
{% include lc/intro.html %}
{% elsif info.carpentry == "ds" %}
{% include ds/intro.md %}
{% remote_include {{lesson_meta}}/description.md %}
{% endif %}

{% comment %}
LOCATION

This block displays the address and links to maps showing directions
if the latitude and longitude of the workshop have been set.  You
can use https://itouchmap.com/latlong.html to find the lat/long of an
address.
{% endcomment %}
{% assign begin_address = info.address | slice: 0, 4 | downcase  %}
{% if info.address == "online" %}
{% assign online = "true_private" %}
{% elsif begin_address contains "http" %}
{% assign online = "true_public" %}
{% else %}
{% assign online = "false" %}
{% endif %}
{% if info.latitude and info.longitude and online == "false" %}
<p id="where">
  <strong>Where:</strong>
  {{info.address}}.
  Get directions with
  <a href="//www.openstreetmap.org/?mlat={{info.latitude}}&mlon={{info.longitude}}&zoom=16">OpenStreetMap</a>
  or
  <a href="//maps.google.com/maps?q={{info.latitude}},{{info.longitude}}">Google Maps</a>.
</p>
{% elsif online == "true_public" %}
<p id="where">
  <strong>Where:</strong>
  online at <a href="{{info.address}}">{{info.address}}</a>.
  If you need a password or other information to access the training,
  the instructor will pass it on to you before the workshop.
</p>
{% elsif online == "true_private" %}
<p id="where">
  <strong>Where:</strong> This training will take place online.
  The instructors will provide you with the information you will need to connect to this meeting.
</p>
{% endif %}

{% comment %}
DATE

This block displays the date and time.
{% endcomment %}
{% if info.humandate %}
<p id="when">
  <strong>When:</strong>
  {% if info.humantime %}
    {{info.humandate}}, {{info.humantime}}.
  {% else %}
    {{info.humandate}}.
  {% endif %}
</p>
{% endif %}

{% comment %}
SPECIAL REQUIREMENTS

Modify the block below if there are any special requirements.
{% endcomment %}
<p id="requirements">
  <strong>Requirements:</strong>
  {% if online == "false" %}
    Participants must bring a laptop with a
    Mac, Linux, or Windows operating system (not a tablet, Chromebook, etc.) that they have administrative privileges on.
  {% else %}
    Participants must have access to a computer with a
    Mac, Linux, or Windows operating system (not a tablet, Chromebook, etc.) that they have administrative privileges on.
  {% endif %}
  They should have a few specific software packages installed (listed <a href="#setup">below</a>).
</p>



{% comment %}
CONTACT EMAIL ADDRESS

Display the contact email address set in the configuration file.
{% endcomment %}
<p id="contact">
  <strong>Contact:</strong>
  Please email
  {% if site.email %}
  {% for email in site.email %}
  {% if forloop.last and site.email.size > 1 %}
  or
  {% else %}
  {% unless forloop.first %}
  ,
  {% endunless %}
  {% endif %}
  <a href='mailto:{{email}}'>{{email}}</a>
  {% endfor %}
  {% else %}
  to-be-announced
  {% endif %}
  for more information.
</p>



<hr/>

{% comment%}
CODE OF CONDUCT
{% endcomment %}
<h2 id="code-of-conduct">Code of Conduct</h2>

{% if info.carpentry == "ds" %}
<p>
Participants are expected to follow these guidelines:
<ul>
  <li>Use welcoming and inclusive language</li>
  <li>Be respectful of different viewpoints and experiences</li>
  <li>Gracefully accept constructive criticism</li>
  <li>Focus on what is best for the community</li>
  <li>Show courtesy and respect towards other community members</li>
</ul>
</p>
{% else %}
<p>
Everyone who participates in Carpentries activities is required to conform to the <a href="https://docs.carpentries.org/topic_folders/policies/code-of-conduct.html">Code of Conduct</a>. This document also outlines how to report an incident if needed.
</p>

<p class="text-center">
  <a href="https://goo.gl/forms/KoUfO53Za3apOuOK2">
    <button type="button" class="btn btn-info">Report a Code of Conduct Incident</button>
  </a>
</p>
<hr/>
{% endif %}


{% comment %}
Collaborative Notes

If you want to use an Etherpad, go to

https://pad.carpentries.org/YYYY-MM-DD-site

where 'YYYY-MM-DD-site' is the identifier for your workshop,
e.g., '2015-06-10-esu'.

Note we also have a CodiMD (the open-source version of HackMD)
available at https://codimd.carpentries.org
{% endcomment %}
{% if page.collaborative_notes %}
<h2 id="collaborative_notes">Collaborative Notes</h2>

<p>
We will use this <a href="{{ page.collaborative_notes }}">collaborative document</a> for chatting, taking notes, and sharing URLs and bits of code.
</p>
<hr/>
{% endif %}


{% comment %}
SCHEDULE

Show the workshop's schedule.

Small changes to the schedule can be made by modifying the
`schedule.html` found in the `_includes` folder for your
workshop type (`swc`, `lc`, or `dc`). Edit the items and
times in the table to match your plans. You may also want to
change 'Day 1' and 'Day 2' to be actual dates or days of the
week.

For larger changes, a blank template for a 4-day workshop
(useful for online teaching for instance) can be found in
`_includes/custom-schedule.html`. Add the times, and what
you will be teaching to this file. You may also want to add
rows to the table if you wish to break down the schedule
further. To use this custom schedule here, replace the block
of code below the Schedule `<h2>` header below with
`{% include custom-schedule.html %}`.
{% endcomment %}

{% if info.carpentry == "ds" %}
<h2 id="syllabus">Syllabus</h2>

Machine learning concepts
- What is machine learning?
- Different types of machine learning
- Big picture of machine learning models
- General pipeline

The predictive modeling pipeline
- Tabular data exploration
- Fitting a scikit-learn model on numerical data
- Handling categorical data

Machine learning algorithms
- Intuitions on linear models
- Overview of machine learning algorithms

Machine learning best practices
- Data hygiene
- Correct evaluation
- How to keep your machine learning project organised

Applying machine learning on LISS dataset:
- How to prepare real-world data for machine learning?
- Take the first steps in predicting fertility using LISS data

{% endif %}

<h2 id="schedule">Schedule</h2>
<div class="row">
  <div class="col-md-6">
   <h3>Day 1</h3>
    <table class="table table-striped">
      <tr> <th>local time</th> <th>what</th></tr>
      <tr> <td>09:00</td>  <td>Welcome and icebreaker</td> </tr>
      <tr> <td>09:15</td>  <td>Introduction to machine learning</td> </tr>
      <tr> <td>10:00</td>  <td>Break  </td> </tr>
      <tr> <td>10:10</td>  <td>Tabular data exploration </td> </tr>
      <tr> <td>11:00</td>  <td>Break</td> </tr>
      <tr> <td>11:10</td>  <td>First model with scikit-learn  </td> </tr>
      <tr> <td>12:00</td>  <td>Lunch Break</td> </tr>
      <tr> <td>13:00</td>  <td>Fitting a scikit-learn model on numerical data</td> </tr>
      <tr> <td>14:00</td>  <td>Working with numerical data </td> </tr>
      <tr> <td>14:20</td>  <td>Break</td> </tr>
      <tr> <td>14:30</td>  <td>Intuition on linear models</td> </tr>
      <tr> <td>15:00</td>  <td>Handling categorical data</td> </tr>
      <tr> <td>15:50</td>  <td>Break</td> </tr>
      <tr> <td>16:00</td>  <td>Guest lecture</td> </tr>
      <tr> <td>17:00</td>  <td>END</td> </tr>
    </table>
  </div>
  <div class="col-md-6">
    <h3>Day 2</h3>
    <table class="table table-striped">
      <tr> <th>local time</th> <th>what</th></tr>
      <tr> <td>09:00</td>  <td>Welcome and recap</td> </tr>
      <tr> <td>09:15</td>  <td>Fertility prediction assignment</td> </tr>
      <tr> <td>10:00</td>  <td>Break</td> </tr>
      <tr> <td>10:10</td>  <td>Fertility prediction assignment </td> </tr>
      <tr> <td>11:00</td>  <td>Break </td> </tr>
      <tr> <td>11:10</td>  <td>Fertility prediction assignment </td> </tr>
      <tr> <td>12:00</td>  <td>Lunch Break</td> </tr>
      <tr> <td>13:00</td>  <td>Machine learning best practices and next steps</td> </tr>
      <tr> <td>14:00</td>  <td>Fertility prediction assignment</td> </tr>
      <tr> <td>14:20</td>  <td>Break</td> </tr>
      <tr> <td>14:30</td>  <td>Hand in first solution to benchmark <br> Q&A </td> </tr>
      <tr> <td>15:30</td>  <td>Wrap-up & Post-workshop Survey</td> </tr>
      <tr> <td>15:50</td>  <td>Break</td> </tr>
      <tr> <td>16:00</td>  <td>Guest lecture</td> </tr>
      <tr> <td>17:00</td>  <td>END</td> </tr>
    </table>
  </div>
</div>

<p><b>All times in the schedule are in the CEST timezone.</b></p>


<hr/>


{% comment %}
SETUP

Delete irrelevant sections from the setup instructions.  Each
section is inside a 'div' without any classes to make the beginning
and end easier to find.

This is the other place where people frequently make mistakes, so
please preview your site before committing, and make sure to run
'tools/check' as well.
{% endcomment %}

<h2 id="setup">Setup</h2>

<p>
  To participate in
  {% if info.carpentry == "swc" %}
  a Software Carpentry
  {% elsif info.carpentry == "dc" %}
  a Data Carpentry
  {% elsif info.carpentry == "lc" %}
  a Library Carpentry
  {% else %}
  this
  {% endif %}
  workshop,
  you will need access to software as described below.
  In addition, you will need an up-to-date web browser.
</p>
<p>
  We maintain a list of common issues that occur during installation as a reference for instructors
  that may be useful on the
  <a href = "{{site.swc_github}}/workshop-template/wiki/Configuration-Problems-and-Solutions">Configuration Problems and Solutions wiki page</a>.
</p>

{% comment %}
These are the installation instructions for the tools used
during the workshop.
{% endcomment %}

<h3 id="software-setup">Software setup</h3>

{% if info.carpentry == "swc" %}
{% include swc/setup.html %}
{% elsif info.carpentry == "dc" %}
{% include dc/setup.html %}
{% elsif info.carpentry == "lc" %}
{% include lc/setup.html %}
{% elsif info.carpentry == "ds" %}
{% capture content %}
{% remote_include {{lesson_meta}}/setup.md %}
{% endcapture %}
{% if content contains "/setup.md" %}
  {% capture setup %}
  {% remote_include https://raw.githubusercontent.com/{{content | strip}} %}
  {% endcapture %}
  {{ setup | split: "---" | last}}
{% else %}
  {{ content }}
{% endif %}
{% elsif info.carpentry == "pilot" %}
Please check the "Setup" page of
[the lesson site]({{ site.lesson_site }}) for instructions to follow
to obtain the software and data you will need to follow the lesson.
{% endif %}

{% comment %}
For online workshops, the section below provides:
- installation instructions for the Zoom client
- recommendations for setting up Learners' workspace so they can follow along
  the instructions and the videoconferencing

If you do not use Zoom for your online workshop, edit the file
`_includes/install_instructions/videoconferencing.html`
to include the relevant installation instrucctions.
{% endcomment %}
{% if online != "false" %}
{% include install_instructions/videoconferencing.html %}
{% endif %}

