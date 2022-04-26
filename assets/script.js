
function checkOverflow(el) {
   var curOverf = el.style.overflow;

   if (!curOverf || curOverf === "visible")
      el.style.overflow = "hidden";

   var isOverflowing = el.clientWidth < el.scrollWidth
      || el.clientHeight < el.scrollHeight;

   el.style.overflow = curOverf;

   return isOverflowing;
}


function setBounds(imageList, value){
   for (var i = 0; i < imageList.length; i++) {
      imageList[i].style.height = value + "px";
      imageList[i].style.width = value + "px";
   };
}

function checkModal() {
   try{
   var modalContent = document.getElementById("modal_content");
   var modalBody = document.getElementById("modal_body");
   var isSliderDisabled = !document.getElementById("modal_manual_picture_size_switch").checked
   var sliderValue = document.getElementById("modal_slider").value
   } catch(e){
      
   }
   if (modalContent) {
      var modalImages = document.getElementsByClassName("modal_images");
      if (isSliderDisabled) {
         currentBounds = 256
         while (checkOverflow(modalBody)) {
            setBounds(modalImages, currentBounds - 8)
            currentBounds -= 8;
         }
      } else {
         setBounds(modalImages, sliderValue)
      }
   }
}


currentBounds = 256
let refreshID = setInterval(checkModal, 250);

