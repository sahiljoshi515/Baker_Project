import Image from "next/image";
import fs from "fs";

import axios from 'axios';
import FormData from 'form-data';

function MyButton(){
  return(
    <button>
        Example Button
    </button>
  );
}

// function handleClick() {
//   alert('You clicked me!');
// }

// const [file, uploadFile] = useState(null)

// //when upload button clicked
// function handleSubmit(){
//     console.log(file[0].name)
//     const formdata = new FormData();
//     formdata.append(
//       "file",
//       file[0],
//     )
//     axios.post("/uploadfile", {
//       file:formdata}, {
//         "Content-Type": "multipart/form-data",
//       })
//           .then(function (response) {
//             console.log(response); //"dear user, please check etc..."
//           });
      
//   }

// // this is when file has been selected
// function handleChange(e){
//   uploadFile(e.target.files); //store uploaded file in "file" variable with useState
// }

export default function Home() {
  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
        <div className="flex flex-col gap-4 items-center">
          <a>Search Tool Provided by Baker Institute for Political Science</a>
          <MyButton />
        </div>
    </div>
  );
}
