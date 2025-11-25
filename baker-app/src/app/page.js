'use client'

import { useRouter } from "next/navigation";
import Link from "next/link"

import Image from "next/image";

import axios from 'axios';
import FormData from 'form-data';

function Search(){
  return(
    <button>
        Get Documents
    </button>
  );
}

function Process(){
  return(
    <button>
        Process Documents
    </button>
  );
}

function handleClick() {
  console.log('You clicked me!');
}

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

  const router = useRouter();
  const handleClick = () => {
    router.push("/search")
  };

  return (
    <div className="grid grid-rows-[0px_1fr_300px] items-center justify-items-center min-h-screen p-8 pb-20 sm:p-20 font-[family-name:var(--font-geist-sans)] font-medium">
        <a>Search Tool Provided by Baker Institute for Political Science</a>
        {/* flex flex-col gap-4 items-center*/}
        <div className="flex items-center justify-items-center">
          <button onClick = {handleClick} className="bg-gray-500 text-white p-2 rounded text-sm w-auto">
                Get Documents
          </button>          
          <button onClick = {handleClick} className="bg-green-500 text-white p-2 ml-6 rounded text-sm w-auto">
                Process Documents
          </button>
        </div>
    </div>
  );
}
