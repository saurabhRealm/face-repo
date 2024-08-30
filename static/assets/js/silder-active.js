(function ($) {
	"use strict";

	////brand-slider
	var tp_brand_slide = new Swiper(".tp-brand-active", {
		loop: true,
		freemode: true,
		slidesPerView: 'auto',
		spaceBetween: 30,
		centeredSlides: true,
		allowTouchMove: false,
		speed: 4000,
		autoplay: {
		  delay: 1,
		  disableOnInteraction: true,
		},
	});

	////text-slider
	var tp_text_slide = new Swiper(".tp-text-active", {
		loop: true,
		freemode: true,
		slidesPerView: 'auto',
		spaceBetween: 0,
		centeredSlides: true,
		allowTouchMove: false,
		speed: 20000,
		autoplay: {
		  delay: 1,
		  disableOnInteraction: true,
		},
	});

	////text-slider
	var tp_text_slide = new Swiper(".tp-text-slider-3-active", {
		loop: true,
		freemode: true,
		slidesPerView: 'auto',
		spaceBetween: 55,
		centeredSlides: true,
		allowTouchMove: false,
		speed: 10000,
		autoplay: {
		  delay: 1,
		  disableOnInteraction: true,
		},
	});

	//project-slider
	var showcase = new Swiper('.tp-project-active', {
		slidesPerView: 4,
		loop: true,
		autoplay: false,
		arrow: false,
		spaceBetween: 15,
		speed: 1000,
		breakpoints: {
			'1600': {
				slidesPerView: 4,
			},
			'1400': {
				slidesPerView: 3,
			},
			'1200': {
				slidesPerView: 3,
			},
			'992': {
				slidesPerView: 2,
			},
			'768': {
				slidesPerView: 2,
			},
			'576': {
				slidesPerView: 1,
			},
			'0': {
				slidesPerView: 1,
			},
		},
		a11y: false,

	});

	//testimonial-slider
	var showcase = new Swiper('.tp-testimonial-active', {
		slidesPerView: 1,
		loop: true,
		autoplay: true,
		arrow: false,
		spaceBetween: 0,
		speed: 1000,
		pagination: {
			el: ".tp-testimonial-dot",
			clickable: true,
		},

		a11y: false,
	});

	//testimonial-slider
	var showcase = new Swiper('.tp-slider-active', {
		slidesPerView: 1,
		loop: true,
		autoplay: true,
		arrow: false,
		spaceBetween: 0,
		speed: 1000,
		effect:'fade',
		pagination: {
			el: ".tp-slider-dot",
			clickable: true,
		},

		a11y: false,
	});

	//testimonial-slider
	var showcase = new Swiper('.tp-testimonial-3-active', {
		slidesPerView: 1,
		loop: true,
		autoplay: true,
		arrow: false,
		spaceBetween: 50,
		speed: 1000,
		pagination: {
			el: ".tp-testimonial-3-dot",
			clickable: true,
		},

		a11y: false,
	});

	//testimonial-slider
	var showcase = new Swiper('.tp-testimonial-4-active', {
		slidesPerView: 5,
		loop: true,
		autoplay: true,
		arrow: false,
		spaceBetween: 25,
		speed: 1000,
		breakpoints: {
			'1600': {
				slidesPerView: 5,
			},
			'1400': {
				slidesPerView: 4,
			},
			'1200': {
				slidesPerView: 3,
			},
			'992': {
				slidesPerView: 2,
			},
			'768': {
				slidesPerView: 2,
			},
			'576': {
				slidesPerView: 1,
			},
			'0': {
				slidesPerView: 1,
			},
		},
		pagination: {
			el: ".tp-testimonial-4-dot",
			clickable: true,
		},

	});
	
	//brand-slider
	var brand_4 = new Swiper('.tp-brand-4-active', {
		loop: true,
		freemode: true,
		slidesPerView: 'auto',
		spaceBetween: 105,
		centeredSlides: true,
		allowTouchMove: false,
		speed: 2000,
		breakpoints: {
			'768': {
				spaceBetween: 105,
			},
			'576': {
				spaceBetween: 50,
			},
			'0': {
				spaceBetween: 50,
			},
		},
		autoplay: {
		  delay: 1,
		  disableOnInteraction: true,
		},

	});

	//brand-slider
	var brand_4 = new Swiper('.tp-brand-5-active', {
		loop: true,
		freemode: true,
		slidesPerView: 'auto',
		spaceBetween: 90,
		centeredSlides: true,
		allowTouchMove: false,
		speed: 2000,
		breakpoints: {
			'768': {
				spaceBetween: 90,
			},
			'576': {
				spaceBetween: 50,
			},
			'0': {
				spaceBetween: 50,
			},
		},
		autoplay: {
		  delay: 1,
		  disableOnInteraction: true,
		},

	});
    
	//portfolio-slider
	var portfolio_1 = new Swiper('.pr-details-slider-active', {
		slidesPerView: 3,
		loop: true,
		autoplay: true,
		arrow: false,
		spaceBetween: 70,
		speed: 1000,
		centeredSlides: true,
		breakpoints: {
			'1200': {
				slidesPerView: 3,
			},
			'992': {
				slidesPerView: 1,
			},
			'768': {
				slidesPerView: 1,
			},
			'576': {
				slidesPerView: 1,
			},
			'0': {
				slidesPerView: 1,
			},
		},
		pagination: {
			el: ".pr-details-dot",
			clickable: true,
		},

	});


	////testimonial-slider
	var testimonial = new Swiper('.postbox-slider-active', {
		slidesPerView: 1,
		loop: true,
		autoplay: false,
		arrow: false,
		spaceBetween: 0,
		speed: 1000,
		effect: 'fade',
		breakpoints: {
			'1400': {
				slidesPerView: 1,
			}
		},
		a11y: false,

		// Navigation arrows
		navigation: {
			prevEl: '.postbox-arrow-prev',
			nextEl: '.postbox-arrow-next',
		},

	});



	
	

})(jQuery);