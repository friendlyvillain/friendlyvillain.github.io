---
# the default layout is 'page'
icon: fas fa-info-circle
order: 4
---

포스팅 시, 주로 사용하는 markdown 구문 정리 

## POST header

```yaml
---
title: TITLE
date: YYYY-MM-DD HH:MM:SS +/-TTTT
author: AUTHOR_INFO
categories: [TOP_CATEGORIE, SUB_CATEGORIE]
tags: [TAG]     # TAG names should always be lowercase
pin: # true or flase (defalut: false)
---
```

## Lists

### Ordered list

1. 1st item
2. 2nd item
3. 3rd item

### Unordered list

- Chapter
  + Section
    * Paragraph

### ToDo list

- [ ] Job
  + [x] Step 1
  + [x] Step 2
  + [ ] Step 3

### Description list

Sun
: the star around which the earth orbits

Moon
: the natural satellite of the earth, visible by reflected light from the sun

## Block Quote

> This line shows the _block quote_.

> An example showing the `tip` type prompt.
{: .prompt-tip }

> An example showing the `info` type prompt.
{: .prompt-info }

> An example showing the `warning` type prompt.
{: .prompt-warning }

> An example showing the `danger` type prompt.
{: .prompt-danger }


## Mathematics

Add post header **`math: true`**.

```yaml
---
title: TITLE
...
math: true
---
```
Grammar is similar with LaTex. 

## References

[Chripy Demo Page](https://chirpy.cotes.page/)