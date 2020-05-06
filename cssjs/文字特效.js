onload = function() {
        var click_cnt = 0;
        var $html = document.getElementsByTagName("html")[0];
        var $body = document.getElementsByTagName("body")[0];
        $html.onclick = function(e) {
            var $elem = document.createElement("b");
            $elem.style.color = "#2c2c2c";/*点击效果颜色*/
            $elem.style.zIndex = 9999;
            $elem.style.position = "absolute";
            $elem.style.select = "none";
            var x = e.pageX;
            var y = e.pageY;
            $elem.style.left = (x - 10) + "px";
            $elem.style.top = (y - 20) + "px";
            clearInterval(anim);
            switch (++click_cnt) {
                case 10:
                    $elem.innerText = "<(￣3￣)> 表！";
                    break;
                case 20:
                    $elem.innerText = "ʅ（´◔౪◔）ʃ";
                    break;
                case 30:
                    $elem.innerText = "(๑•́ ₃ •̀๑)";
                    break;
                case 40:
                    $elem.innerText = "(๑•̀_•́๑)";
                    break;
                case 50:
                    $elem.innerText = "（￣へ￣）";
                    break;
                case 60:
                    $elem.innerText = "(╯°口°)╯(┴—┴";
                    break;
                case 70:
                    $elem.innerText = "Ψ(￣∀￣)Ψ";
                    break;
                case 80:
                    $elem.innerText = "╮(｡>口<｡)╭";
                    break;
                case 90:
                    $elem.innerText = "( ง ᵒ̌皿ᵒ̌)ง⁼³₌₃";
                    break;
                case 100:
                    $elem.innerText = "(ノへ￣、)";
                    break;
                case 101:
                    $elem.innerText = "o(￣ε￣*)";
                    break;
                case 102:/*此处可以按照上面的格式添加表情*/
                case 103:
                case 104:
                case 105:
                    $elem.innerText = "(ꐦ°᷄д°᷅)";
                    break;
                default:
                    $elem.innerText = "٩(●´৺`●)وbiu";
                    break;
            }
            $elem.style.fontSize = Math.random() * 10 + 8 + "px";
            var increase = 0;
            var anim;
            setTimeout(function() {
                anim = setInterval(function() {
                    if (++increase == 150) {
                        clearInterval(anim);
                        $body.removeChild($elem);
                    }
                    $elem.style.top = y - 20 - increase + "px";
                    $elem.style.opacity = (150 - increase) / 120;
                }, 8);
            }, 70);
            $body.appendChild($elem);
        };
    };