---
layout: archive
title: "Blog"
permalink: /blog/
author_profile: true
---

{% include base_path %}

<p class="blog-note"><em>The write-ups, literature follow-ups and interactive figures here are made with an LLM, which I prompt and whose output I read and correct. I would not have had the time to produce them otherwise.</em></p>

{% for post in site.posts %}
  {% include archive-single.html %}
{% endfor %}
