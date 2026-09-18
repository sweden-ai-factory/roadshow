selector_to_html = {"a[href=\"#instructor-s-guide\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Instructor\u2019s guide<a class=\"headerlink\" href=\"#instructor-s-guide\" title=\"Link to this heading\">\u00b6</a></h1><h2>Why we teach this lesson<a class=\"headerlink\" href=\"#why-we-teach-this-lesson\" title=\"Link to this heading\">\u00b6</a></h2>", "a[href=\"#interesting-questions-you-might-get\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Interesting questions you might get<a class=\"headerlink\" href=\"#interesting-questions-you-might-get\" title=\"Link to this heading\">\u00b6</a></h2>", "a[href=\"#preparing-exercises\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Preparing exercises<a class=\"headerlink\" href=\"#preparing-exercises\" title=\"Link to this heading\">\u00b6</a></h2><p>e.g. what to do the day before to set up common repositories.</p>", "a[href=\"#other-practical-aspects\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Other practical aspects<a class=\"headerlink\" href=\"#other-practical-aspects\" title=\"Link to this heading\">\u00b6</a></h2><p>Before teaching the lesson, instructors should decide how far to go into the\nattention calculation. The important distinction is between:</p>", "a[href=\"#why-we-teach-this-lesson\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Why we teach this lesson<a class=\"headerlink\" href=\"#why-we-teach-this-lesson\" title=\"Link to this heading\">\u00b6</a></h2>", "a[href=\"#intended-learning-outcomes\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Intended learning outcomes<a class=\"headerlink\" href=\"#intended-learning-outcomes\" title=\"Link to this heading\">\u00b6</a></h2>", "a[href=\"#timing\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Timing<a class=\"headerlink\" href=\"#timing\" title=\"Link to this heading\">\u00b6</a></h2>", "a[href=\"#typical-pitfalls\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Typical pitfalls<a class=\"headerlink\" href=\"#typical-pitfalls\" title=\"Link to this heading\">\u00b6</a></h2>"}
skip_classes = ["headerlink", "sd-stretched-link"]

window.onload = function () {
    for (const [select, tip_html] of Object.entries(selector_to_html)) {
        const links = document.querySelectorAll(` ${select}`);
        for (const link of links) {
            if (skip_classes.some(c => link.classList.contains(c))) {
                continue;
            }

            tippy(link, {
                content: tip_html,
                allowHTML: true,
                arrow: true,
                placement: 'auto-start', maxWidth: 500, interactive: false,

            });
        };
    };
    console.log("tippy tips loaded!");
};
