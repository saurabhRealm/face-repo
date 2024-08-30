/***************************************************
==================== JS INDEX ======================
****************************************************
01. PreLoader Js
02. Mobile Menu Js
03. Common Js
04. Menu Controls JS
05. Offcanvas Js
06. Search Js
07. cartmini Js
08. filter
09. Body overlay Js
10. Sticky Header Js
11. Theme Settings Js
12. Nice Select Js
13. Smooth Scroll Js
14. Slider Activation Area Start
15. Masonary Js
16. Wow Js
17. Counter Js
18. InHover Active Js
19. Line Animation Js
20. Video Play Js
21. Password Toggle Js
****************************************************/

(function ($) {
	"use strict";

	var windowOn = $(window);

	// 01. PreLoader Js//
	windowOn.on('load', function () {
		$("#loading").fadeOut(500);
	});

	// mobile menu 
	var tpMenuWrap = $('.tp-mobile-menu-active > ul').clone();
	var tpSideMenu = $('.tp-offcanvas-menu nav');
	tpSideMenu.append(tpMenuWrap);
	if ($(tpSideMenu).find('.tp-submenu, .tp-mega-menu').length != 0) {
		$(tpSideMenu).find('.tp-submenu, .tp-mega-menu').parent().append
			('<button class="tp-menu-close"><i class="far fa-chevron-right"></i></button>');
	}
	var sideMenuList =
		$('.tp-offcanvas-menu nav > ul > li button.tp-menu-close, .tp-offcanvas-menu nav > ul li.has-dropdown > a');
	$(sideMenuList).on('click', function (e) {
		e.preventDefault();
		if (!($(this).parent().hasClass('active'))) {
			$(this).parent().addClass('active');
			$(this).siblings('.tp-submenu, .tp-mega-menu').slideDown();
		} else {
			$(this).siblings('.tp-submenu, .tp-mega-menu').slideUp();
			$(this).parent().removeClass('active');
		}
	});

	// 07. Sticky Header Js
	windowOn.on('scroll', function () {
		var scroll = windowOn.scrollTop();
		if (scroll < 20) {
			$("#header-sticky").removeClass("header-sticky");
		} else {
			$("#header-sticky").addClass("header-sticky");
		}
	});


	// offcanvas
	$(".tp-offcanvas-open-btn").on("click", function () {
		$(".tp-offcanvas-area").addClass("opened");
		$(".body-overlay").addClass("opened");
	});
	$(".tp-offcanvas-close-btn").on("click", function () {
		$(".tp-offcanvas-area").removeClass("opened");
		$(".body-overlay").removeClass("opened");
	});

	// // 09. Body overlay Js
	$(".body-overlay").on("click", function () {
		$(".tp-offcanvas-area").removeClass("opened");
		$(".body-overlay").removeClass("opened");
	});


	// 03. back-to-top Js//
	function back_to_top() {
		var btn = $('#back_to_top');
		var btn_wrapper = $('.back-to-top-wrapper');

		windowOn.scroll(function () {
			if (windowOn.scrollTop() > 300) {
				btn_wrapper.addClass('back-to-top-btn-show');
			} else {
				btn_wrapper.removeClass('back-to-top-btn-show');
			}
		});

		btn.on('click', function (e) {
			e.preventDefault();
			$('html, body').animate({ scrollTop: 0 }, '300');
		});
	}
	back_to_top();


	// 13. Smooth Scroll Js
	function smoothSctoll() {
		$('.smooth a').on('click', function (event) {
			var target = $(this.getAttribute('href'));
			if (target.length) {
				event.preventDefault();
				$('html, body').stop().animate({
					scrollTop: target.offset().top - 120
				}, 1500);
			}
		});
	}
	smoothSctoll();



	// 02. Wow Js//
	new WOW().init();

	// 03. Common Js//
	$("[data-background]").each(function () {
		$(this).css("background-image", "url( " + $(this).attr("data-background") + "  )");
	});

	$("[data-width]").each(function () {
		$(this).css("width", $(this).attr("data-width"));
	});

	$("[data-bg-color]").each(function () {
		$(this).css("background-color", $(this).attr("data-bg-color"));
	});

	$("[data-text-color]").each(function () {
		$(this).css("color", $(this).attr("data-text-color"));
	});

	// 04. Nice Select Js//
	$('select').niceSelect();


	// 05. Masonary Js//
	$('.grid').imagesLoaded(function () {

		var $grid = $('.grid').isotope({
			itemSelector: '.grid-item',
			percentPosition: true,
			layoutMode: 'fitRows',
			masonry: {
				columnWidth: 1,
			},

		});
		// filter items on button click
		$('.masonary-menu').on('click', 'button', function () {
			var filterValue = $(this).attr('data-filter');
			$grid.isotope({
				filter: filterValue,
				animationOptions: {
					duration: 1000,
					easing: "linear",
					queue: false
				}
			});

		});
		//for menu active class
		$('.masonary-menu button').on('click', function (event) {
			$(this).siblings('.active').removeClass('active');
			$(this).addClass('active');
			event.preventDefault();
		});

	});

	$('.grid-2').imagesLoaded(function () {
		var $grid = $('.grid-2').isotope({
			itemSelector: '.grid-item',
			percentPosition: true,
			masonry: {
				columnWidth: '.grid-item',
			}
		});

	});


	// 06. magnificPopup img view //
	$('.popup-image').magnificPopup({
		type: 'image',
		gallery: {
			enabled: true
		}
	});

	// 07. magnificPopup video view //
	$(".popup-video").magnificPopup({
		type: "iframe",
	});

	// 08. Counter Js //
	new PureCounter();
	new PureCounter({
		filesizing: true,
		selector: ".filesizecount",
		pulse: 2,
	});


	// 09. smooth scroll wrapper //
	if ($('#smooth-wrapper').length && $('#smooth-content').length) {
		gsap.registerPlugin(ScrollTrigger, ScrollSmoother, TweenMax, ScrollToPlugin);

		gsap.config({
			nullTargetWarn: false,
		});

		let smoother = ScrollSmoother.create({
			smooth: 1.35,
			effects: true,
			smoothTouch: 0.1,
			normalizeScroll: false,
			ignoreMobileResize: true,
		});
	}

	// price toggle animation
	function tabtable_active_1() {
		const $elements = {
			monthly: $("#tp-nav-monthly"),
			yearly: $("#tp-nav-yearly"),
			switcher: $("#tp-switcher-input"),
			tabMonthly: $("#tp-tab-monthly"),
			tabYearly: $("#tp-tab-yearly")
		};

		const setActive = isMonthly => {
			$elements.switcher.prop("checked", isMonthly);
			$elements.monthly.toggleClass("is-active", isMonthly);
			$elements.yearly.toggleClass("is-active", !isMonthly);
			$elements.tabMonthly.toggleClass("tp-tab-hide", !isMonthly);
			$elements.tabYearly.toggleClass("tp-tab-hide", isMonthly);
		};

		[$elements.monthly, $elements.yearly].forEach($el =>
			$el.on("click", () => setActive($el.is($elements.monthly)))
		);

		$elements.switcher.on("click", () =>
			setActive(!$elements.monthly.hasClass("is-active"))
		);
	}
	if ($("#tp-nav-monthly").length) tabtable_active_1();


	// button hover animation
	$('.tp-hover-btn').on('mouseenter', function (e) {
		var x = e.pageX - $(this).offset().left;
		var y = e.pageY - $(this).offset().top;

		$(this).find('.tp-btn-circle-dot').css({
			top: y,
			left: x
		});
	});

	$('.tp-hover-btn').on('mouseout', function (e) {
		var x = e.pageX - $(this).offset().left;
		var y = e.pageY - $(this).offset().top;

		$(this).find('.tp-btn-circle-dot').css({
			top: y,
			left: x
		});
	});

	var hoverBtns = gsap.utils.toArray(".tp-hover-btn-wrapper");

	const hoverBtnItem = gsap.utils.toArray(".tp-hover-btn-item");
	hoverBtns.forEach((btn, i) => {
		$(btn).mousemove(function (e) {
			callParallax(e);
		});
		function callParallax(e) {
			parallaxIt(e, hoverBtnItem[i], 60);
		}

		function parallaxIt(e, target, movement) {
			var $this = $(btn);
			var relX = e.pageX - $this.offset().left;
			var relY = e.pageY - $this.offset().top;

			gsap.to(target, 1, {
				x: ((relX - $this.width() / 2) / $this.width()) * movement,
				y: ((relY - $this.height() / 2) / $this.height()) * movement,
				ease: Power2.easeOut,
			});
		}
		$(btn).mouseleave(function (e) {
			gsap.to(hoverBtnItem[i], 1, {
				x: 0,
				y: 0,
				ease: Power2.easeOut,
			});
		});
	});

	// service tab
	if ($('#lineMarker').length > 0) {

		function tp_tab_bg() {
			var marker = document.querySelector('#lineMarker');
			var item = document.querySelectorAll('.tp-marker-tab ul li button');
			var itemActive = document.querySelector('.tp-marker-tab ul li .nav-links.active');

			function indicator(e) {
				marker.style.left = e.offsetLeft + "px";
				marker.style.width = e.offsetWidth + "px";
			}


			item.forEach(link => {
				link.addEventListener('click', (e) => {
					indicator(e.target);
				});
			});

			var activeNav = $('.tp-marker-tab ul li .nav-links.active');
			var activewidth = $(activeNav).width();
			var activePadLeft = parseFloat($(activeNav).css('padding-left'));
			var activePadRight = parseFloat($(activeNav).css('padding-right'));
			var totalWidth = activewidth + activePadLeft + activePadRight;

			var precedingAnchorWidth = anchorWidthCounter();


			$(marker).css('display', 'block');

			$(marker).css('width', totalWidth);

			function anchorWidthCounter() {
				var anchorWidths = 0;
				var a;
				var aWidth;
				var aPadLeft;
				var aPadRight;
				var aTotalWidth;
				$('.tp-marker-tab ul li button').each(function (index, elem) {
					var activeTest = $(elem).hasClass('active');
					marker.style.left = elem.offsetLeft + "px";
					if (activeTest) {

						return false;
					}

					a = $(elem).find('button');
					aWidth = a.width();
					aPadLeft = parseFloat(a.css('padding-left'));
					aPadRight = parseFloat(a.css('padding-right'));
					aTotalWidth = aWidth + aPadLeft + aPadRight;

					anchorWidths = anchorWidths + aTotalWidth;

				});

				return anchorWidths;
			}
		}
		tp_tab_bg();
	}

	if ($('#lineMarker-2').length > 0) {

		function tp_tab_line_2() {
			var marker = document.querySelector('#lineMarker-2');
			var item = document.querySelectorAll('.tp-marker-tab-2 ul li button');
			var itemActive = document.querySelector('.tp-marker-tab-2 ul li .nav-links.active');

			function indicator(e) {
				marker.style.left = e.offsetLeft + "px";
				marker.style.width = e.offsetWidth + "px";
			}


			item.forEach(link => {
				link.addEventListener('click', (e) => {
					indicator(e.target);
				});
			});

			var activeNav = $('.tp-marker-tab-2 ul li .nav-links.active');
			var activewidth = $(activeNav).width();
			var activePadLeft = parseFloat($(activeNav).css('padding-left'));
			var activePadRight = parseFloat($(activeNav).css('padding-right'));
			var totalWidth = activewidth + activePadLeft + activePadRight;

			var precedingAnchorWidth = anchorWidthCounter();


			$(marker).css('display', 'block');

			$(marker).css('width', totalWidth);

			function anchorWidthCounter() {
				var anchorWidths = 0;
				var a;
				var aWidth;
				var aPadLeft;
				var aPadRight;
				var aTotalWidth;
				$('.tp-marker-tab-2 ul li button').each(function (index, elem) {
					var activeTest = $(elem).hasClass('active');
					marker.style.left = elem.offsetLeft + "px";
					if (activeTest) {

						return false;
					}

					a = $(elem).find('button');
					aWidth = a.width();
					aPadLeft = parseFloat(a.css('padding-left'));
					aPadRight = parseFloat(a.css('padding-right'));
					aTotalWidth = aWidth + aPadLeft + aPadRight;

					anchorWidths = anchorWidths + aTotalWidth;

				});

				return anchorWidths;
			}
		}
		tp_tab_line_2();
	}


	// faq-class-active
	$('.tp-faq-style-2 .accordion-items').on("click", function () {
		$(this).addClass('active').siblings().removeClass('active');
	});

	// about-class-active
	$('.tp-about-5-list ul li a').on("mouseenter", function () {
		$(".tp-about-5-list ul li a").removeClass("active");
		$(this).addClass("active");
	});

	// header-class-active
	if ($("#tp-header-search").length > 0) {
		window.addEventListener('click', function (e) {
			if (document.getElementById('tp-header-search').contains(e.target)) {
				$(".tp-header-search-input").toggleClass("active");
			}
			else {
				$(".tp-header-search-input").removeClass("active");
			}
		});
	}

	//Reveal Animation 
	const anim_reveal = document.querySelectorAll(".tp_reveal_anim");
	anim_reveal.forEach(areveal => {
		const duration_value = areveal.getAttribute("data-duration") || 1.5;
		const onscroll_value = areveal.getAttribute("data-on-scroll") || 1;
		const stagger_value = areveal.getAttribute("data-stagger") || 0.02;
		const data_delay = areveal.getAttribute("data-delay") || 0.05;
		areveal.split = new SplitText(areveal, {
			type: "lines,words,chars",
			linesClass: "tp-reveal-line"
		});
		const animationProps = {
			duration: duration_value,
			delay: data_delay,
			ease: "circ.out",
			y: 200,
			stagger: stagger_value,
			opacity: 0
		};
		if (onscroll_value == 1) {
			animationProps.scrollTrigger = {
				trigger: areveal,
				start: 'top 85%',
				end: 'bottom 15%',
			};
		}
		areveal.anim = gsap.from(areveal.split.chars, animationProps);
	});

	// fade-class-active
	if ($(".tp_fade_anim").length > 0) {
		gsap.utils.toArray(".tp_fade_anim").forEach((item) => {
			let tp_fade_offset = item.getAttribute("data-fade-offset") || 40,
				tp_duration_value = item.getAttribute("data-duration") || 0.75,
				tp_fade_direction = item.getAttribute("data-fade-from") || "bottom",
				tp_onscroll_value = item.getAttribute("data-on-scroll") || 1,
				tp_delay_value = item.getAttribute("data-delay") || 0.15,
				tp_ease_value = item.getAttribute("data-ease") || "power2.out",
				tp_anim_setting = {
					opacity: 0,
					ease: tp_ease_value,
					duration: tp_duration_value,
					delay: tp_delay_value,
					x: (tp_fade_direction == "left" ? -tp_fade_offset : (tp_fade_direction == "right" ? tp_fade_offset : 0)),
					y: (tp_fade_direction == "top" ? -tp_fade_offset : (tp_fade_direction == "bottom" ? tp_fade_offset : 0)),
				};

			if (tp_onscroll_value == 1) {
				tp_anim_setting.scrollTrigger = {
					trigger: item,
					start: 'top 85%',
				};
			}
			gsap.from(item, tp_anim_setting);
		});
	}

	// text-animation
	$(document).ready(function () {
		var st = $(".tp-split-text");
		if (st.length === 0) return;

		gsap.registerPlugin(SplitText);
		st.each(function (index, el) {
			el.split = new SplitText(el, {
				type: "lines,words,chars",
				linesClass: "tp-split-line"
			});

			gsap.set(el, { perspective: 400 });
			var animationProps = { opacity: 0 };

			if ($(el).hasClass('tp-split-right')) animationProps.x = "50";
			if ($(el).hasClass('tp-split-left')) animationProps.x = "-50";
			if ($(el).hasClass('tp-split-up')) animationProps.y = "80";
			if ($(el).hasClass('tp-split-down')) animationProps.y = "-80";

			gsap.set(el.split.chars, animationProps);

			el.anim = gsap.to(el.split.chars, {
				scrollTrigger: { trigger: el, start: "top 90%" },
				x: "0",
				y: "0",
				rotateX: "0",
				scale: 1,
				opacity: 1,
				duration: 0.4,
				stagger: 0.02,
			});
		});
	});






})(jQuery);